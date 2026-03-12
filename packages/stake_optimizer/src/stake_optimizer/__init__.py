"""Stake Optimizer"""

from stake_optimizer.adapters import (
    adapt_eqy_sh_out,
    adapt_queue_delays,
    adapt_reward_rate,
)
from stake_optimizer.analytics import max_drawdown_n
from stake_optimizer.schemas import Schemas

__all__ = [
    "Schemas",
    "adapt_eqy_sh_out",
    "adapt_queue_delays",
    "adapt_reward_rate",
    "max_drawdown_n",
]

"""
Treat the following as context, not things to implement just yet:

We have a balancing act here. We have an ETF with ETH, and we want to optimize our
policy for selecting the amount of ETH to stake. The benefit of staking is obviously
the staking reward. However, once we initiate a stake (unstake), the amount we allocate
becomes locked for a certain period of time (the entry [exit] queue delay). This delay
varies but is known upon decision time. Also, staked ETH is obviously locked and cannot
be used to service redemptions. While in the queues, we are exposed to redemption risk,
which we can proxy with the max drawdown in shares outstanding over the delay period.

If enough shares are redeemed (via create/redeem) to overwhelm our liquid ETH holdings,
we will have to resort to financing by borrowing ETH at a borrow cost which is variable
and substantial and correlated with queue delays. Redemptions must be serviced within a
T+2 delay (we should treat this as configurable in code).
"""
