# put your strategies here.
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


class RSIStrategy(BaseNode):
    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        RSI动量策略，结合了多种交易信号进行分析:
        1. RSI超买超卖信号
        2. RSI背离信号
        3. RSI趋势强度
        4. RSI波动率分析
        5. 多时间周期RSI综合分析
        """

        data = state['data']
        data['name'] = "RSIStrategy"

        tickers = data.get("tickers", [])
        intervals = data.get("intervals", [])

        # 为每个ticker初始化分析结果
        rsi_analysis = {}
        for ticker in tickers:
            rsi_analysis[ticker] = {}

        # 使用加权集成方法组合所有信号
        strategy_weights = {
            "rsi_overbought_oversold": 0.05,
            "rsi_divergence": 0.05,
            "rsi_trend_strength": 0.05,
            "rsi_volatility": 0.05,
            "momentum": 0.80,
        }

        for ticker in tickers:
            for interval in intervals:
                df = data.get(f"{ticker}_{interval.value}", pd.DataFrame())
                
                # 计算RSI相关信号
                trend_signals = calculate_trend_signals(df)
                momentum_signals = calculate_momentum_signals(df)
                mean_reversion_signals = calculate_mean_reversion_signals(df)
                volatility_signals = calculate_volatility_signals(df)
                
                # 将RSI特定信号与其他信号结合
                combined_signal = weighted_signal_combination(
                    {
                        "rsi_overbought_oversold": mean_reversion_signals,  # RSI超买超卖主要是均值回归信号
                        "rsi_divergence": trend_signals,  # RSI背离通常与趋势信号相关
                        "rsi_trend_strength": momentum_signals,  # RSI趋势强度与动量相关
                        "rsi_volatility": volatility_signals,  # RSI波动率分析
                        "momentum": momentum_signals,  # 额外的动量信号
                    },
                    strategy_weights,
                )

                # 为该ticker生成详细分析报告
                rsi_analysis[ticker][interval.value] = {
                    "signal": combined_signal["signal"],
                    "confidence": round(combined_signal["confidence"] * 100),
                    "strategy_signals": {
                        "rsi_overbought_oversold": {
                            "signal": mean_reversion_signals["signal"],
                            "confidence": round(mean_reversion_signals["confidence"] * 100),
                            "metrics": normalize_pandas(mean_reversion_signals["metrics"]),
                        },
                        "rsi_divergence": {
                            "signal": trend_signals["signal"],
                            "confidence": round(trend_signals["confidence"] * 100),
                            "metrics": normalize_pandas(trend_signals["metrics"]),
                        },
                        "rsi_trend_strength": {
                            "signal": momentum_signals["signal"],
                            "confidence": round(momentum_signals["confidence"] * 100),
                            "metrics": normalize_pandas(momentum_signals["metrics"]),
                        },
                        "rsi_volatility": {
                            "signal": volatility_signals["signal"],
                            "confidence": round(volatility_signals["confidence"] * 100),
                            "metrics": normalize_pandas(volatility_signals["metrics"]),
                        },
                        "momentum": {
                            "signal": momentum_signals["signal"],
                            "confidence": round(momentum_signals["confidence"] * 100),
                            "metrics": normalize_pandas(momentum_signals["metrics"]),
                        },
                    },
                }

        # 创建RSI分析师消息
        message = HumanMessage(
            content=json.dumps(rsi_analysis),
            name="technical_analyst_agent",
        )

        if state["metadata"]["show_reasoning"]:
            show_agent_reasoning(rsi_analysis, "RSI Analyst")

        # 将信号添加到analyst_signals列表
        state["data"]["analyst_signals"]["technical_analyst_agent"] = rsi_analysis

        return {
            "messages": [message],
            "data": data,
        }
