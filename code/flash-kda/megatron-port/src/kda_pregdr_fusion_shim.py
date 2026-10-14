# Copyright (c) 2026, tech_record contributors.
#
# M14-1: runtime wiring of the KDA-specific *native* fused pre-GDR (M13-1 forward +
# M13-2/M13-3 backward, ``src/kda_pregdr_bwd.py::kda_pre_gdr_fused_native``) into the
# ``gdn_pre_gated_delta_rule_fusion`` path of ``KimiDeltaAttention`` -- **without editing the
# Megatron-LM tree** (a constraint every prior megatron-port round kept).
#
# Upstream, KDA deliberately rejects that flag in two places (see ``docs/14`` §2):
#
#   * config guard  ``transformer_config.py:1936-1938``  (``TransformerConfig.__post_init__``)
#   * forward guard ``kda.py:399-402``                    (``KimiDeltaAttention.forward``)
#
# Both are *the same structural judgement* ("the GDN streamed-fusion kernel cannot be reused
# for KDA, so refuse").  This shim keeps that judgement but replaces "refuse" with "dispatch to
# the KDA-specific fused kernels": it patches the two guard sites at runtime so a config with
# ``gdn_pre_gated_delta_rule_fusion=True`` + ``experimental_attention_variant="kda"`` builds and
# runs, and the layer's ``pre_gated_delta_rule`` call is served by the native fused family.
#
# This is the mirror of what a real tree edit would look like -- replace the ``raise`` with a
# branch that calls a KDA ``pre_gated_delta_rule_fused`` -- but done as a monkeypatch so the
# numerical result can be anchored end-to-end first.
#
# The shim is dependency-free at import time (it never imports Megatron); the caller passes the
# (already imported) classes and the fused callable.

import types

_STATE = {"installed": False, "orig_post": None, "orig_forward": None,
          "config_cls": None, "kimi_cls": None}


def install(kimi_cls, config_cls, fused_fn):
    """Lift both ``gdn_pre_gated_delta_rule_fusion`` guards for KDA at runtime.

    ``kimi_cls``  : ``megatron.core.ssm.gated_delta_net.KimiDeltaAttention``
    ``config_cls``: ``megatron.core.transformer.TransformerConfig``
    ``fused_fn``  : ``(module, qkvfg, beta, batch, seq_len) -> 8-tuple`` authentic autograd
                    callable, i.e. ``kda_pregdr_bwd.kda_pre_gdr_fused_native``.
    """
    if _STATE["installed"]:
        return
    _STATE.update(installed=True, config_cls=config_cls, kimi_cls=kimi_cls)

    # ---- config guard: let ``__post_init__`` run with the flag masked, then restore it ----
    orig_post = config_cls.__post_init__
    _STATE["orig_post"] = orig_post

    def _post(self):
        saved = getattr(self, "gdn_pre_gated_delta_rule_fusion", False)
        variant = getattr(self, "experimental_attention_variant", None)
        if saved and variant == "kda":
            object.__setattr__(self, "gdn_pre_gated_delta_rule_fusion", False)
            try:
                orig_post(self)
            finally:
                object.__setattr__(self, "gdn_pre_gated_delta_rule_fusion", saved)
        else:
            orig_post(self)

    config_cls.__post_init__ = _post

    # ---- forward guard: when the flag is set, swap in the fused pre-GDR for this call ----
    orig_forward = kimi_cls.forward
    _STATE["orig_forward"] = orig_forward

    def _fused_method(self_mod, qkvfg, beta, batch, seq_len, *args, **kwargs):
        # The eager ``pre_gated_delta_rule`` also takes cp / varlen args; the KDA-specific
        # fused family here is SBHD cp=1 (GVA factor 1), so they are ignored on purpose.
        return fused_fn(self_mod, qkvfg, beta, batch, seq_len)

    def forward(self, *args, **kwargs):
        if getattr(self, "gdn_pre_gated_delta_rule_fusion", False):
            saved_flag = self.gdn_pre_gated_delta_rule_fusion
            saved_pgdr = self.pre_gated_delta_rule
            self.gdn_pre_gated_delta_rule_fusion = False          # passes the :399 guard
            self.pre_gated_delta_rule = types.MethodType(_fused_method, self)
            try:
                return orig_forward(self, *args, **kwargs)
            finally:
                self.pre_gated_delta_rule = saved_pgdr
                self.gdn_pre_gated_delta_rule_fusion = saved_flag
        return orig_forward(self, *args, **kwargs)

    kimi_cls.forward = forward


def uninstall():
    """Undo :func:`install` (restores the original guards)."""
    if not _STATE["installed"]:
        return
    _STATE["config_cls"].__post_init__ = _STATE["orig_post"]
    _STATE["kimi_cls"].forward = _STATE["orig_forward"]
    _STATE["installed"] = False
    # ``orig_post`` is a bound-ish function already; re-applying repeatedly is idempotent.
