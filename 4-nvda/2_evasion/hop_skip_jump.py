import torch
import numpy as np

class HopSkipJump:
    def __init__(self, model):
        self.model = model

    def adversarial_satisfactory(self, samples, target, clip_min, clip_max):
        """Check success.
        """
        samples = torch.clamp(samples, clip_min, clip_max)
        preds = self.model(samples).argmax(dim=1)
        result = preds != target
        # print(preds, target, result)
        return result

    def compute_delta(self, current_sample, original_sample, theta, norm, clip_min, clip_max, input_shape, curr_iter=None):
        """Compute the delta parameter.
        """
        # Note: This is a bit different from the original paper, instead we keep those that are
        # implemented in the original source code of the authors
        if curr_iter == 0:
            return 0.1 * (clip_max - clip_min)

        if norm == 2:
            dist = torch.norm(original_sample - current_sample)
            delta = torch.sqrt(torch.prod(torch.tensor(input_shape))) * theta * dist
        else:
            dist = torch.max(torch.abs(original_sample - current_sample))
            delta = torch.prod(torch.tensor(input_shape)) * theta * dist

        return delta

    def binary_search(self, current_sample, original_sample, target, norm, clip_min, clip_max, threshold):
        """Binary search to approach the boundary.
        """
        # First set upper and lower bounds as well as the threshold for the binary search
        if norm == 2:
            (upper_bound, lower_bound) = (1, 0)

        else:
            (upper_bound, lower_bound) = (
                torch.maximum(abs(original_sample - current_sample)),
                0,
            )

            if threshold is None:
                threshold = torch.minimum(upper_bound * theta, theta)

        # Then start the binary search
        while (upper_bound - lower_bound) > threshold:
            # Interpolation point
            alpha = (upper_bound + lower_bound) / 2.0
            interpolated_sample = self.interpolate(
                current_sample=current_sample,
                original_sample=original_sample,
                alpha=alpha,
                norm=norm,
            )

            # Update upper_bound and lower_bound
            satisfied = self.adversarial_satisfactory(
                samples=interpolated_sample,
                target=target,
                clip_min=clip_min,
                clip_max=clip_max,
            )

            lower_bound = torch.where(satisfied == 0, alpha, lower_bound)
            upper_bound = torch.where(satisfied == 1, alpha, upper_bound)

        result = self.interpolate(
            current_sample=current_sample,
            original_sample=original_sample,
            alpha=upper_bound,
            norm=norm,
        )

        return result

    def compute_update(self, current_sample, num_eval, delta, target, clip_min, clip_max, norm, input_shape):
        """Compute the update in Eq.(14). See paper.
        """
        # Generate random noise
        rnd_noise_shape = [num_eval] + list(input_shape)
        if norm == 2:
            rnd_noise = torch.randn(*rnd_noise_shape)
        else:
            rnd_noise = torch.rand(rnd_noise_shape) * 2 - 1


        # Normalize random noise to fit into the range of input data
        rnd_noise = rnd_noise / torch.sqrt(
            torch.sum(
                rnd_noise ** 2,
                axis=tuple(range(len(rnd_noise_shape)))[1:],
                keepdims=True,
            )
        )
        eval_samples = torch.clamp(current_sample + delta * rnd_noise, clip_min, clip_max)
        rnd_noise = (eval_samples - current_sample) / delta

        satisfied = self.adversarial_satisfactory(
            samples=eval_samples, target=target, clip_min=clip_min, clip_max=clip_max
        )
        f_val = 2 * satisfied.reshape([num_eval] + [1] * len(input_shape)) - 1.0
        # f_val = f_val.astype(ART_NUMPY_DTYPE)

        if torch.mean(f_val) == 1.0:
            grad = torch.mean(rnd_noise, axis=0)
        elif torch.mean(f_val) == -1.0:
            grad = -torch.mean(rnd_noise, axis=0)
        else:
            f_val -= torch.mean(f_val)
            grad = torch.mean(f_val * rnd_noise, axis=0)

        # Compute update
        if norm == 2:
            result = grad / torch.norm(grad)
        else:
            result = torch.sign(grad)

        return result


    def interpolate(self, current_sample, original_sample, alpha, norm):
        """Interpolate a new sample based on the original and the current samples.
        """
        if norm == 2:
            result = (1 - alpha) * original_sample + alpha * current_sample
        else:
            result = torch.clamp(current_sample, original_sample - alpha, original_sample + alpha)

        return result


    def run(self, x, y=None):
        params = {
            "targeted": False,
            "norm": 2,
            "max_iter": 100,
            "max_eval": 10000,
            "init_eval": 10,
            "init_size": 10,
            "curr_iter": 0,
            "batch_size": 1,
            "verbose": True,
            "clip_min": -1,
            "clip_max": 1
        }

        final_results = []

        # Set binary search threshold
        if params["norm"] == 2:
            theta = 0.01 / np.sqrt(np.prod(x.shape))
        else:
            theta = 0.01 / np.prod(x.shape)

        # Prediction from the original image
        y = self.model(x)

        input_shape = x.squeeze(0).shape

        # Generate the adversarial samples
        for ind, val in enumerate(x):
            curr_iter = 0

            # First, create an initial adversarial sample

            generator = torch.Generator().manual_seed(0)
            initial_sample = None

            for _ in range(params["init_size"]):
                y_p = y[ind].argmax()
                random_img = torch.FloatTensor(x.shape).uniform_(params["clip_min"], params["clip_max"], generator=generator)
                random_class = self.model(random_img).argmax()

                original_sample=x[ind].unsqueeze(0)

                if random_class != y_p:
                    # Binary search to reduce the l2 distance to the original image
                    random_img = self.binary_search(
                        current_sample=random_img,
                        original_sample=original_sample,
                        target=y_p,
                        norm=params["norm"],
                        clip_min=params["clip_min"],
                        clip_max=params["clip_max"],
                        threshold=theta
                    )

                    initial_sample = random_img, y_p

                    print("[+] Found misclassified image!")
                    break

            # Main loop to wander around the boundary
            current_sample = initial_sample
            original_sample = original_sample

            for _ in range(params["max_iter"]):
                # First compute delta
                delta = self.compute_delta(
                    current_sample=current_sample[0],
                    original_sample=original_sample,
                    theta=theta,
                    norm=params["norm"],
                    clip_min=params["clip_min"],
                    clip_max=params["clip_max"],
                    input_shape=input_shape,
                    curr_iter=curr_iter
                )

                current_sample = self.binary_search(
                    current_sample=current_sample[0],
                    original_sample=original_sample,
                    target=y_p,
                    norm=params["norm"],
                    clip_min=params["clip_min"],
                    clip_max=params["clip_max"],
                    threshold=theta,
                )

                num_eval = min(int(params["init_eval"] * torch.sqrt(torch.tensor(curr_iter) + 1)), params["max_eval"])

                update = self.compute_update(
                    current_sample=current_sample,
                    num_eval=num_eval,
                    delta=delta,
                    target=y_p,
                    norm=params["norm"],
                    clip_min=params["clip_min"],
                    clip_max=params["clip_max"],
                    input_shape=input_shape
                )

                # Finally run step size search by first computing epsilon
                if params["norm"] == 2:
                    dist = torch.norm(original_sample - current_sample)
                else:
                    dist = torch.max(abs(original_sample - current_sample))

                epsilon = 2.0 * dist / torch.sqrt(torch.tensor(curr_iter + 1))
                success = False

                while not success:
                    epsilon /= 2.0
                    potential_sample = current_sample + epsilon * update
                    success = self.adversarial_satisfactory(
                        samples=potential_sample,
                        target=y_p,
                        clip_min=params["clip_min"],
                        clip_max=params["clip_max"],
                    )

                # Update current sample
                current_sample = torch.clamp(potential_sample, params["clip_min"], params["clip_max"])

                # Update current iteration
                curr_iter += 1

                final_results.append(
                    (current_sample, curr_iter, dist.item()))
                print(curr_iter, dist.item())

        return final_results