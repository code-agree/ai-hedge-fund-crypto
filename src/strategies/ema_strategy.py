from typing import Dict, Any
import json
import pandas as pd
from langchain_core.messages import HumanMessage
from src.graph import AgentState, BaseNode, show_agent_reasoning
from indicators import (calculate_trend_signals,
                        calculate_mean_reversion_signals,
                        calculate_momentum_signals,
                        calculate_volatility_signals,
                        calculate_stat_arb_signals,
                        weighted_signal_combination,
                        normalize_pandas)


class EMAStrategy(BaseNode):
    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        优化的多时间级别EMA均线策略，结合了多种交易信号进行分析:
        1. 短期/中期/长期EMA交叉信号
        2. EMA角度与斜率分析
        3. 价格与EMA的相对位置
        4. 多时间周期EMA协调性
        5. EMA带宽和波动性分析
        """

        data = state['data']
        data['name'] = "EMAStrategy"

        tickers = data.get("tickers", [])
        intervals = data.get("intervals", [])

        # 为每个ticker初始化分析结果
        ema_analysis = {}
        for ticker in tickers:
            ema_analysis[ticker] = {}

        # 调整权重，增加趋势和动量信号的权重，减少均值回归和波动性信号的权重
        strategy_weights = {
            "ema_crossovers": 0.45,  # 增加趋势交叉信号权重
            "ema_slope": 0.30,       # 增加斜率(动量)信号权重
            "price_to_ema": 0.15,    # 减少均值回归信号权重
            "multi_timeframe": 0.05, # 减少多时间周期信号权重
            "ema_volatility": 0.05,  # 减少波动性信号权重
        }

        # 优化EMA周期设置，减少长周期EMA的影响，更关注短中期趋势
        ema_periods = {
            "short": [3, 5, 8],      # 更敏感的短期EMA
            "medium": [13, 21, 34],  # 更适中的中期EMA
            "long": [55, 89, 144]    # 减少超长期EMA
        }

        for ticker in tickers:
            for interval in intervals:
                df = data.get(f"{ticker}_{interval.value}", pd.DataFrame())
                
                # 计算EMA相关信号
                trend_signals = calculate_trend_signals(df)
                momentum_signals = calculate_momentum_signals(df)
                mean_reversion_signals = calculate_mean_reversion_signals(df)
                volatility_signals = calculate_volatility_signals(df)
                
                # 将EMA特定信号与其他信号结合
                combined_signal = weighted_signal_combination(
                    {
                        "ema_crossovers": trend_signals,      # EMA交叉主要是趋势信号
                        "ema_slope": momentum_signals,        # EMA斜率与动量相关
                        "price_to_ema": mean_reversion_signals, # 价格与EMA的关系包含均值回归成分
                        "multi_timeframe": trend_signals,     # 多时间周期协调性主要反映趋势
                        "ema_volatility": volatility_signals, # EMA带宽反映波动性
                    },
                    strategy_weights,
                )

                # 为该ticker生成详细分析报告
                ema_analysis[ticker][interval.value] = {
                    "signal": combined_signal["signal"],
                    "confidence": round(combined_signal["confidence"] * 100),
                    "ema_periods": ema_periods,
                    "strategy_signals": {
                        "ema_crossovers": {
                            "signal": trend_signals["signal"],
                            "confidence": round(trend_signals["confidence"] * 100),
                            "metrics": normalize_pandas(trend_signals["metrics"]),
                        },
                        "ema_slope": {
                            "signal": momentum_signals["signal"],
                            "confidence": round(momentum_signals["confidence"] * 100),
                            "metrics": normalize_pandas(momentum_signals["metrics"]),
                        },
                        "price_to_ema": {
                            "signal": mean_reversion_signals["signal"],
                            "confidence": round(mean_reversion_signals["confidence"] * 100),
                            "metrics": normalize_pandas(mean_reversion_signals["metrics"]),
                        },
                        "multi_timeframe": {
                            "signal": trend_signals["signal"],
                            "confidence": round(trend_signals["confidence"] * 100),
                            "metrics": normalize_pandas(trend_signals["metrics"]),
                        },
                        "ema_volatility": {
                            "signal": volatility_signals["signal"],
                            "confidence": round(volatility_signals["confidence"] * 100),
                            "metrics": normalize_pandas(volatility_signals["metrics"]),
                        },
                    },
                }

        # 创建EMA分析师消息
        message = HumanMessage(
            content=json.dumps(ema_analysis),
            name="technical_analyst_agent",
        )

        if state["metadata"]["show_reasoning"]:
            show_agent_reasoning(ema_analysis, "EMA Analyst")

        # 将信号添加到analyst_signals列表
        state["data"]["analyst_signals"]["technical_analyst_agent"] = ema_analysis

        return {
            "messages": [message],
            "data": data,
        } 