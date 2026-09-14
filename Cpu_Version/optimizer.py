from __future__ import annotations

import torch


class AttentionMLPWheelOptimizer(
    torch.optim.Optimizer
):

    def __init__(
        self,
        params,
        base_infinitesimal=2e-3,
        projective_ceiling=6.0,
        weight_decay=1e-2,
        alpha=0.05,
    ):

        defaults = dict(
            base_infinitesimal=base_infinitesimal,
            projective_ceiling=projective_ceiling,
            weight_decay=weight_decay,
            alpha=alpha,
        )

        super().__init__(
            params,
            defaults,
        )

    @torch.no_grad()
    def step(self, closure=None):

        loss = None

        if closure is not None:

            with torch.enable_grad():
                loss = closure()

        global_abs = None
        global_numel = 0

        for group in self.param_groups:

            params_with_grad = [
                p
                for p in group["params"]
                if p.grad is not None
            ]

            for p in params_with_grad:

                value = p.grad.abs().sum()

                if global_abs is None:
                    global_abs = value
                else:
                    global_abs += value

                global_numel += p.grad.numel()

        if (
            global_abs is None
            or global_numel == 0
        ):
            return loss

        global_mean_grad = (
            global_abs / global_numel
        )

        global_mean_grad = (
            global_mean_grad + 1e-8
        )

        for group in self.param_groups:

            eps = group[
                "base_infinitesimal"
            ]

            ceiling = group[
                "projective_ceiling"
            ]

            weight_decay = group[
                "weight_decay"
            ]

            alpha = group["alpha"]

            params_with_grad = [
                p
                for p in group["params"]
                if p.grad is not None
            ]

            if len(params_with_grad) == 0:
                continue

            total_abs = torch.zeros(
                1,
                device=params_with_grad[0].device,
                dtype=params_with_grad[0].dtype,
            )

            total_numel = 0

            for p in params_with_grad:

                total_abs += (
                    p.grad.abs().sum()
                )

                total_numel += p.grad.numel()

            local_mean_grad = (
                total_abs / total_numel
            )

            local_mean_grad = (
                local_mean_grad + 1e-8
            )

            mean_grad = (
                (1.0 - alpha)
                * local_mean_grad
                + alpha
                * global_mean_grad
            )

            mean_grad = (
                mean_grad + 1e-8
            )

            for p in params_with_grad:

                grad = p.grad

                abs_grad = grad.abs()

                scaled_grad = (
                    abs_grad / mean_grad
                )

                wheel_magnitude = (
                    scaled_grad
                    / (
                        1.0
                        + scaled_grad / ceiling
                    )
                )

                update = (
                    grad.sign()
                    * wheel_magnitude
                    * eps
                )

                p.data.add_(
                    update,
                    alpha=-1.0,
                )

                if weight_decay != 0.0:

                    p.data.mul_(
                        1.0
                        - eps * weight_decay
                    )

        return loss