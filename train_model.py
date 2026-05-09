import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def main():
    # 读取数据：年份 → 销售额
    df = pd.read_csv("sales_data.csv")
    X = df[["year"]]
    y = df["sales"]

    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X, y)

    # 保存模型
    joblib.dump(model, "sales_model.pkl")

    # 保存结果
    with open("model_summary.txt", "w") as f:
        f.write(f"模型系数: {model.coef_[0]}\n")
        f.write(f"模型截距: {model.intercept_}\n")
        f.write("✅ 销售额预测模型训练完成！\n")

    print("✅ 模型训练成功！")

if __name__ == "__main__":
    main()
