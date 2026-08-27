"""The MACE loop: a weighted Mann iteration over a set of agents.

An agent is a callable taking and returning a torch tensor of one fixed shape
on one fixed device (a reconstruction volume), so a GPU run stays on the GPU
end to end.  With two agents -- a forward-model proximal map and a denoiser --
equal weights, and rho = 1/2, the iteration is PnP-ADMM / Douglas-Rachford;
the general form below also covers multi-agent setups such as multi-slice
fusion (one forward agent plus one denoiser per slice orientation).

The iteration, with mu the agent weights (summing to 1) and rho in (0, 1):

    X_i   = F_i(w_i)                          # evaluate every agent
    z     = sum_i mu_i (2 X_i - w_i)
    w_i  += 2 rho (z - X_i)                   # Mann-averaged (2G - I)(2F - I)
    x_bar = sum_i mu_i X_i                    # consensus estimate

At a fixed point every X_i equals the consensus, so the per-iteration spread
max_i ||X_i - x_bar|| / ||x_bar|| measures how far the agents are from
agreeing, and is the natural convergence trace.
"""

import torch


def _norm(x):
    # Plain sum-of-squares norm: exact enough at these sizes on every backend.
    return float(torch.sqrt(torch.sum(x * x)))


def mace(agents, x0, mu=None, rho=0.5, num_iterations=30, callback=None):
    """Run the MACE fixed-point iteration and return the consensus estimate.

    Args:
        agents: list of callables, each mapping a volume tensor to a volume
            tensor of the same shape and device.
        x0 (tensor): initial volume; every agent input starts here.
        mu (list of float, optional): agent weights summing to 1.
            Defaults to equal weights.
        rho (float, optional): Mann averaging parameter in (0, 1).  Smaller is
            more damped.  Defaults to 0.5 (the ADMM / Douglas-Rachford value
            for two agents).
        num_iterations (int, optional): outer iterations to run.
        callback (callable, optional): called as callback(iteration, x_bar)
            after each iteration, e.g. to record an error trace.

    Returns:
        (x_bar, info): the final consensus estimate, and a dict of
        per-iteration traces: 'consensus_spread' (max over agents of
        ||X_i - x_bar|| / ||x_bar||) and 'consensus_change'
        (||x_bar - previous x_bar|| / ||x_bar||, with 0 for iteration 0).
    """
    num_agents = len(agents)
    if mu is None:
        mu = [1.0 / num_agents] * num_agents
    if len(mu) != num_agents:
        raise ValueError('mu must have one weight per agent. '
                         f'Got {len(mu)} weights for {num_agents} agents.')

    W = [x0.clone() for _ in agents]
    previous_x_bar = None
    info = {'consensus_spread': [], 'consensus_change': []}
    with torch.no_grad():
        for iteration in range(num_iterations):
            X = [agent(w) for agent, w in zip(agents, W)]
            x_bar = sum(m * x for m, x in zip(mu, X))
            z = sum(m * (2.0 * x - w) for m, x, w in zip(mu, X, W))
            W = [w + 2.0 * rho * (z - x) for w, x in zip(W, X)]

            x_bar_norm = _norm(x_bar)
            spread = max(_norm(x - x_bar) for x in X) / x_bar_norm
            change = (_norm(x_bar - previous_x_bar) / x_bar_norm
                      if previous_x_bar is not None else 0.0)
            info['consensus_spread'].append(spread)
            info['consensus_change'].append(change)
            previous_x_bar = x_bar
            if callback is not None:
                callback(iteration, x_bar)
    return x_bar, info
