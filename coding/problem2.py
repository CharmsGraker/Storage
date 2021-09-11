import gurobipy
import numpy as np
import pandas as pd
from gurobipy import *
from matplotlib import pyplot as plt

from coding.preprocess import getLossMatrix, getSatisfyMatrix

os.chdir(r'D:\2021CUMCM\coding')
filename = './../quiz/C/附件1 近5年402家供应商的相关数据.xlsx'

df = pd.read_excel(filename, sheet_name=0)
sheet2_name = './../quiz/C/附件2 近5年8家转运商的相关数据.xlsx'
df_offer = pd.read_excel(sheet2_name)

# 贮存费与C购价的比例系数
BETA = 0.1
e = 1

# 未来24周
future_step = 24

# 402个供应商
n_providers = 402
n_transfor = 8

TRANSFOR_CAPACITY = 6000

PROVIDER = range(0, n_providers)
TRANSFOR = range(0, n_transfor)
WEEKS = range(0, future_step)

MATERIAL = ['A', 'B', 'C']
produce = {'A': 0.6,
           'B': 0.66,
           'C': 0.72}

provider_type = df['材料分类'].values
# 定义周产量
week_produce_cnt = 18000 # 2.82 * 1e4 / 1.5

# 定义满意率
satisfied = getSatisfyMatrix()  # pd.DataFrame(np.ones((future_step, len(PROVIDER))))

# 定义损耗率

transfor_loss = getLossMatrix()

pay_C = e
pay_A = 1.2 * pay_C
pay_B = 1.1 * pay_C

material_pay = {'A': pay_A,
                'B': pay_B,
                'C': pay_C}
keep_cost = BETA * pay_C

MODEL = gurobipy.Model('l_model')

# 变量是订购量
x_Order = MODEL.addVars(WEEKS, PROVIDER, vtype=gurobipy.GRB.INTEGER, name='x_order')

# 实际接收的材料
received = MODEL.addVars(WEEKS, MATERIAL, vtype=gurobipy.GRB.INTEGER, name='received')
# 转移矩阵,决策矩阵
T = MODEL.addVars(WEEKS, PROVIDER, TRANSFOR, MATERIAL, vtype=gurobipy.GRB.INTEGER, name='transfer')

# 供货量向量
provides = MODEL.addVars(WEEKS, PROVIDER, vtype=gurobipy.GRB.INTEGER, name='provides')

# # 每种材料类别的接收量
# receive = MODEL.addVars(WEEKS, TRANSFOR,vtype=gurobipy.GRB.INTEGER)

# 每周用于生产的消耗量
consum_per_week = MODEL.addVars(WEEKS, MATERIAL, vtype=gurobipy.GRB.INTEGER, name='consum_per_week')

# 定义库存
rest = MODEL.addVars(WEEKS, MATERIAL, vtype=gurobipy.GRB.INTEGER, name='rest')

# 定义每周期望消耗
need = MODEL.addVars(WEEKS, MATERIAL, vtype=gurobipy.GRB.INTEGER, name='need')

# 定义目标变量
storage_cost = MODEL.addVar(name='storage_cost')
loss_cost = MODEL.addVar()
spends = MODEL.addVar(name='spends')
stability = MODEL.addVar()
carry_cost = MODEL.addVar()

# 每周的存贮费
storage_cost_per_week = MODEL.addVars(WEEKS, vtype=gurobipy.GRB.INTEGER, name='storage_cost_per_week')

MODEL.update()

MODEL.setObjectiveN(spends, index=0, priority=1)
MODEL.setObjectiveN(loss_cost, index=1, priority=2)
MODEL.setObjectiveN(sum(storage_cost_per_week), index=2,priority=3)
MODEL.setObjectiveN(carry_cost, index=2,priority=4)

# 订单量非负
MODEL.addConstrs((x_Order[week, i] >= 0 for i in PROVIDER for week in WEEKS), name='cons_orders')

# MODEL.addConstr((sum(x_Order[week, i] for i in PROVIDER for week in WEEKS) <= (future_step+2) * week_produce_cnt), name='cons_orders')

MODEL.addConstr(carry_cost == sum(
    T[week, i, j, _type] * keep_cost for i in PROVIDER for j in TRANSFOR for _type in MATERIAL for week in WEEKS))

# 供应量非负
MODEL.addConstrs((provides[week, i] >= 0 for i in PROVIDER for week in WEEKS), name='cons_provides')

# 接收量非负
MODEL.addConstrs((received[week, _type] >= 0 for week in WEEKS for _type in MATERIAL))

# 剩余量非负
MODEL.addConstrs((rest[week, _type] >= 0 for week in WEEKS for _type in MATERIAL))

# (每周）供应的必须全部转出
MODEL.addConstrs(
    sum(T[week, s, j, _type] for j in TRANSFOR for _type in MATERIAL) <= provides[week, s] for s in PROVIDER for week in
    WEEKS)

# 转运商不能超过其承载能力
MODEL.addConstrs(
    sum(T[week, i, j, _type] for i in PROVIDER for _type in MATERIAL) <= TRANSFOR_CAPACITY for j in TRANSFOR for week in
    WEEKS)

# 库存大于两周产量
MODEL.addConstrs((
    sum(rest[week, _type] / produce[_type] for _type in MATERIAL) >= 2 * week_produce_cnt + 1 for week in WEEKS),
    name='storage_above')

# 订购量与供应量关系
MODEL.addConstrs(provides[week, s] == x_Order[week, s] for s in PROVIDER for week in WEEKS)

# 实际接收量
MODEL.addConstrs(
    received[week, _type] == sum(
        (1 - transfor_loss.at[week, j]) * T[week, i, j, _type] for i in PROVIDER for j in TRANSFOR)
    for _type in
    MATERIAL for week in WEEKS)

# 解释目标函数


# 现存的存贮费
MODEL.addConstrs(
    storage_cost_per_week[week] == sum(rest[week, _type] * keep_cost for _type in MATERIAL) for week in WEEKS)

# 订购费
MODEL.addConstr(spends == sum(x_Order[week, i] * material_pay[provider_type[i]] for i in PROVIDER for week in WEEKS))

# 损耗量,为每一家转运商的损耗量之和
MODEL.addConstr(
    loss_cost == sum(
        sum(transfor_loss.at[week, j] * T[week, i, j, _type] for i in PROVIDER for _type in MATERIAL for week in WEEKS)
        for
        j in
        TRANSFOR))

# 剩余比较难以计算，只能动态规划推的
# 优先消耗C，因为C转化率低，用于制造消耗最快。省去存贮费。


# 能消耗的，不超过当前的总量
# consum_per_week存储消耗的材料数。相除得到能生产的产品数


# # 等于当前的总量 - 应该消耗的量
# MODEL.addConstrs(need[week, 'C'] == week_produce_cnt * produce['C'] for week in WEEKS)

# 开始时没有消耗

##################
# MODEL.addConstrs(
#     consum_per_week[week, _type] >= 0 for _type in MATERIAL
#     for week in WEEKS)
#
# MODEL.addConstrs(
#     consum_per_week[week, _type] <= received[week, _type] + rest[week-1, _type] for _type in MATERIAL for week in WEEKS[1:])
#
# MODEL.addConstrs(
#     consum_per_week[0, _type] <= received[0, _type] for _type in MATERIAL)

#
# # 好像不能使用min函数。只能转换成对应的约束了
# # .................................................
#
#
# # MODEL.addConstrs(
# #     need[week, 'B'] == max(0, week_produce_cnt - consum_per_week[week, 'A'] / produce['A']) * produce['B'] for week in
# #     WEEKS[1:])
#
# #
# # MODEL.addConstrs(
# #     need[week, 'B'] >= 0 for week in
# #     WEEKS)
#
#
# MODEL.addConstrs(
#     need[week, 'B'] == (week_produce_cnt - consum_per_week[week, 'C'] / produce['C']) * produce['B'] for week in
#     WEEKS)
#
# # MODEL.addConstrs(
# #     need[week, 'C'] == max(0, week_produce_cnt - sum(
# #         consum_per_week[week, _type] / produce[_type] for _type in ['A', 'B'])) * produce['C'] for week in
# #     WEEKS)
#
#
# MODEL.addConstrs(
#     need[week, 'A'] >= 0 for week in
#     WEEKS)

# MODEL.addConstrs(
#     need[week, 'A'] == (week_produce_cnt - sum(
#         consum_per_week[week, _type] / produce[_type] for _type in ['B', 'C'])) * produce['A'] for week in
#     WEEKS)

# 开始时没有消耗
# MODEL.addConstrs(
#     consum_per_week[week, _type] == (
#         min(need[week, _type], received[week, _type] + rest[week - 1, _type]) if week >= 1 else 0) for _type in
#     MATERIAL
#     for week in WEEKS)


MODEL.addConstrs(
    consum_per_week[week, _type] <= received[week, _type] + rest[week - 1, _type] for _type in
    MATERIAL for week in WEEKS[1:])

MODEL.addConstrs(
    consum_per_week[0, _type] <= received[0, _type] for _type in
    MATERIAL)

MODEL.addConstrs(
    rest[week, _type] == (
            received[week, _type] + rest[week - 1, _type] - consum_per_week[week, _type]) for
    _type in MATERIAL
    for week in WEEKS[1:])

# 第0周
MODEL.addConstrs(
    rest[0, _type] == (
            received[0, _type] - consum_per_week[0, _type]) for
    _type in MATERIAL)

# 每周必须达到产能
MODEL.addConstrs(
    sum((consum_per_week[week, _type] / produce[_type]) for _type in MATERIAL) >= week_produce_cnt for week in WEEKS)

MODEL.addConstrs(
    sum(1 for x in [T[week, i, j, _type] for j in TRANSFOR for _type in MATERIAL] if x) >= 6 for i in PROVIDER for week
    in WEEKS)


print('init done.')
MODEL.Params.TimeLimit = 100

MODEL.optimize()

# for i in MODEL.getVars():
#     if (i.x):
#         print(i)
MODEL.computeIIS()
MODEL.write("model1.ilp")
#
# # plt.plot()
#
# Order = np.zeros((24,402))
# for i in WEEKS:
#     for j in PROVIDER:
#         Order[i,j] = x_Order[i,j].x
#
# np.savetxt('p2_order.csv',Order)
#
# Provide_real = np.zeros((24,402))
# for i in WEEKS:
#     for j in PROVIDER:
#         Provide_real[i,j] = provides[i,j].x
#
# np.savetxt('p2_provides.csv', Provide_real)

