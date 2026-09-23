import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

API_KEY = "sYOUR_API_KEY"
BASE_URL = "https://fuyao.aicubes.cn"


class DoubleMovingAveragesStrategy:
    def __init__(self, initial_capital=1000000):
        self.initial_capital = initial_capital
        self.data = None
        self.signals = None
        self.portfolio = None
        self.transactions = None

        # 设置双均线的周期
        self.short_window = 5
        self.long_window = 20

    def get_stock_data(self):
        stock_code = "600519.SH"

        print(f"正在从富耀获取{stock_code}数据...")

        url = f"{BASE_URL}/api/a-share/prices/historical"

        headers = {
            "X-api-key": API_KEY
        }

        params = {
            "thscode": stock_code,
            "interval": "1d",
            "start": 1640966400000,
            "end": 1672531200000
        }

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        result = response.json()

        if result.get("code") != 0:
            raise Exception(f"数据获取失败：{result}")

        df = pd.DataFrame(result["data"]["item"])

        df["trade_date"] = pd.to_datetime(
            df["date_ms"],
            unit="ms"
        )

        df = df.rename(columns={
            "open_price": "open",
            "high_price": "high",
            "low_price": "low",
            "close_price": "close",
            "volume": "vol"
        })

        df = df[
            [
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "vol"
            ]
        ]

        df = df.sort_values("trade_date")
        df.set_index("trade_date", inplace=True)

        print(f"真实数据获取成功！共{len(df)}条记录")

        return df

    def create_sample_data(self):
        dates = pd.date_range('2022-01-01', '2023-01-01', freq='B')
        dates = dates[dates.dayofweek < 5]
        np.random.seed(42)
        prices = [1900]

        for i in range(1, len(dates)):
            trend = 2 if i < len(dates)//2 else -1.5
            volatility = np.random.normal(trend, 25)
            new_price = prices[-1] + volatility
            new_price = max(new_price, 1500)
            new_price = min(new_price, 2800)
            prices.append(new_price)

        df = pd.DataFrame({
            'open': [p*0.99 + np.random.normal(0,5) for p in prices],
            'high': [p*1.02 + np.random.normal(0,3) for p in prices],
            'low': [p*0.98 + np.random.normal(0,3) for p in prices],
            'close': prices,
            'vol': [np.random.randint(1000000, 5000000) for _ in prices]
        }, index=dates)

        print(f"模拟数据创建成功！共 {len(df)} 个交易日")

        return df

    def calculate_signals(self):
        """计算双均线并生成交易信号"""
        if self.data is None:
            print("请先获取数据！")
            return

        df = self.data.copy()

        # 1. 计算短均线和长均线
        df['short_ma'] = df['close'].rolling(
            window=self.short_window
        ).mean()

        df['long_ma'] = df['close'].rolling(
            window=self.long_window
        ).mean()

        # 2. 生成交易信号
        df['positions'] = 0

        df.loc[
            (df['short_ma'] > df['long_ma']) &
            (df['short_ma'].shift(1) <= df['long_ma'].shift(1)),
            'positions'
        ] = 1

        df.loc[
            (df['short_ma'] < df['long_ma']) &
            (df['short_ma'].shift(1) >= df['long_ma'].shift(1)),
            'positions'
        ] = -1

        self.portfolio = df

        return df

    def run_backtest(self):
        """执行模拟交易并计算每日资产"""
        if self.portfolio is None:
            print("请先计算信号！")
            return

        cash = self.initial_capital
        shares = 0
        portfolio_values = []
        cash_history = []
        shares_history = []
        transactions = []

        # 遍历每日数据
        for i, (date, row) in enumerate(self.portfolio.iterrows()):

            close_price = row['close']
            position_change = row['positions']

            # 买入逻辑
            if position_change == 1 and cash > close_price:

                shares_to_buy = cash // close_price

                if shares_to_buy > 0:

                    cost = shares_to_buy * close_price
                    cash -= cost
                    shares += shares_to_buy

                    transactions.append({
                        'date': date,
                        'action': 'BUY',
                        'price': close_price,
                        'shares': shares_to_buy,
                        'amount': cost,
                        'cash_after': cash,
                        'shares_after': shares
                    })

                    print(
                        f"{date.strftime('%Y-%m-%d')}: "
                        f"买入 {shares_to_buy}股, "
                        f"价格 {close_price:.2f}"
                    )

            # 卖出逻辑
            elif position_change == -1 and shares > 0:

                proceeds = shares * close_price
                cash += proceeds

                transactions.append({
                    'date': date,
                    'action': 'SELL',
                    'price': close_price,
                    'shares': shares,
                    'amount': proceeds,
                    'cash_after': cash,
                    'shares_after': 0
                })

                print(
                    f"{date.strftime('%Y-%m-%d')}: "
                    f"卖出 {shares}股, "
                    f"价格 {close_price:.2f}"
                )

                shares = 0

            # 记录每日总资产
            total_value = cash + shares * close_price

            portfolio_values.append(total_value)
            cash_history.append(cash)
            shares_history.append(shares)

        self.portfolio['portfolio_value'] = portfolio_values
        self.portfolio['cash'] = cash_history
        self.portfolio['shares'] = shares_history

        self.transactions = (
            pd.DataFrame(transactions)
            if transactions
            else pd.DataFrame()
        )

        print("策略回测完成！")

    def calculate_metrics(self):
        """简单计算绩效指标"""
        if self.portfolio is None or 'portfolio_value' not in self.portfolio.columns:
            print("请先运行回测！")
            return

        df = self.portfolio

        total_return = (
            df['portfolio_value'].iloc[-1]
            / self.initial_capital - 1
        ) * 100

        # 计算最大回撤
        roll_max = df['portfolio_value'].cummax()

        drawdown = (
            df['portfolio_value'] - roll_max
        ) / roll_max

        max_drawdown = drawdown.min() * 100

        print("\n" + "="*30)
        print("      回测绩效报告")
        print("="*30)
        print(f"初始资金: {self.initial_capital:.2f}")
        print(f"期末资金: {df['portfolio_value'].iloc[-1]:.2f}")
        print(f"策略总收益: {total_return:.2f}%")
        print(f"最大回撤: {max_drawdown:.2f}%")
        print(f"交易次数: {len(self.transactions)}")
        print("="*30)

    def plot_results(self):
        """绘图"""
        if self.portfolio is None:
            print("请先运行回测！")
            return

        df = self.portfolio

        plt.figure(figsize=(14, 10))

        # 1. 价格与均线
        plt.subplot(2, 1, 1)

        plt.plot(
            df.index,
            df['close'],
            label='Close Price',
            color='gray',
            alpha=0.6
        )

        plt.plot(
            df.index,
            df['short_ma'],
            label=f'{self.short_window}-day MA',
            color='blue'
        )

        plt.plot(
            df.index,
            df['long_ma'],
            label=f'{self.long_window}-day MA',
            color='red'
        )

        # 标记买卖点
        buys = df[df['positions'] == 1]
        sells = df[df['positions'] == -1]

        plt.scatter(
            buys.index,
            buys['close'],
            marker='^',
            color='green',
            s=100,
            label='Buy Signal'
        )

        plt.scatter(
            sells.index,
            sells['close'],
            marker='v',
            color='red',
            s=100,
            label='Sell Signal'
        )

        plt.title(
            'Double Moving Average Strategy - Price & Signals'
        )

        plt.legend()
        plt.grid(True)

        # 2. 资金曲线
        plt.subplot(2, 1, 2)

        plt.plot(
            df.index,
            df['portfolio_value'],
            label='Strategy Net Value',
            color='orange'
        )

        plt.title('Portfolio Value Over Time')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


# ================= 运行测试 =================

if __name__ == "__main__":

    strategy = DoubleMovingAveragesStrategy(
        initial_capital=1000000
    )

    # 1. 获取真实数据
    strategy.data = strategy.get_stock_data()

    # 2. 计算信号
    print("\n开始计算双均线信号...")
    strategy.calculate_signals()

    # 3. 执行回测
    print("开始执行回测...")
    strategy.run_backtest()

    # 4. 评估和绘图
    strategy.calculate_metrics()
    strategy.plot_results()