"""程序生成的“难题”：长材料 + 多步推理 + 日期 / 数字计算，答案由程序算出，保证正确。

    python -m ajev.data.hard_gen --out data/hard/train.jsonl --n 4500 --seed 1
    python -m ajev.data.hard_gen --out data/hard/dev.jsonl   --n 500  --seed 2

为什么要生成：lora2 / lora3 两轮训练在 JevBench hard 档（0.70–0.71）和长材料题（0.676）上都没有进步。
统计发现训练集里 57% 的题材料只有几十个 token，材料超过 1,500 token 的只占 2.1%；而 JevBench hard 档有 33%。
模型几乎没练过“读一页政策 / 几张表，再一步步推出结论”的题。公开数据集里这类题很少，所以用程序生成：
规则和数据随机组合，答案由代码计算，想要多少有多少，而且一定正确。

五类题（对应 JevBench hard 档考的能力：长政策、多步查找、日期和数字）：

=================  ===================================================  =========================
类别                材料                                                 问题
=================  ===================================================  =========================
refund 退款政策      退货政策（期限、会员延长、例外品类、开封扣费、破损）+ 订单记录    能否退、退多少
invoice 发票对账     采购单、收货单、发票、已付发票清单 + 对账规则（价格容差等）      怎么处理、数量是否都有收货
contract 合同期限    合同（初始期限、自动续约、提前通知天数）+ 可能有修订条款 + 解约通知  某天合同处于什么状态、是否续约过
approval 审批路由    员工表、部门表、汇率表、审批矩阵 + 地区和职级特殊规则           需要谁审批、折算后金额
sla 工单时效         SLA 表（按优先级和客户等级）、客户表、节假日、工单时间线         首次响应是否超时、超时多少
=================  ===================================================  =========================

难度设计：
- 每份材料都有**干扰信息**：不相关的政策条款、别的订单 / 员工 / 工单，答题时必须先找到对的那一条；
- 约 30% 是**长材料**（数十到上百行表格 + 大量条款），对应 1,500–5,000 token；
- 关键规则之间有**优先级**（例如“破损在 7 天内报告，即使是清仓品也全额退款”），必须按顺序判断；
- 每个问题的答案分布做了**均衡**（拒绝采样），避免模型靠“总猜同一个选项”拿分。
"""

from __future__ import annotations

import argparse
import datetime as dt
import random
from collections import Counter
from typing import Callable

from ajev.schema import DEFAULT_NOUL_DESC, NOUL_FALSE, NOUL_TRUE, Decision, Option, state_to_text, write_jsonl

D = dt.date
WEEKDAY_ZH = "一二三四五六日"


def fmt_date(d: D, lang: str) -> str:
    return f"{d.year}年{d.month}月{d.day}日" if lang == "zh" else d.isoformat()


def add_months(d: D, n: int) -> D:
    """日期加 n 个月；目标月份没有这一天时取月末（例如 1 月 31 日 + 1 个月 → 2 月 28/29 日）。"""
    m = d.month - 1 + n
    y, m = d.year + m // 12, m % 12 + 1
    last = (D(y + m // 12, m % 12 + 1, 1) - dt.timedelta(days=1)).day
    return D(y, m, min(d.day, last))


def noul(id_: str, source: str, state: str, instr: str, answer: bool, lang: str, group: str) -> Decision:
    desc = DEFAULT_NOUL_DESC[lang]
    return Decision(id=id_, source=source, type="noul", state=state, instructions=instr,
                    options=[Option(NOUL_TRUE, desc[NOUL_TRUE]), Option(NOUL_FALSE, desc[NOUL_FALSE])],
                    target=[1.0, 0.0] if answer else [0.0, 1.0], lang=lang, group=group, meta={"gold": str(answer)})


def choice(id_: str, source: str, state: str, instr: str, options: list[tuple[str, str]], gold: str, lang: str,
           group: str) -> Decision:
    return Decision(id=id_, source=source, type="choice", state=state, instructions=instr,
                    options=[Option(n, d) for n, d in options],
                    target=[1.0 if n == gold else 0.0 for n, _ in options], lang=lang, group=group, meta={"gold": gold})


def score(id_: str, source: str, state: str, instr: str, levels: list[str], gold: int, lang: str, group: str) -> Decision:
    """打分题：levels 从低到高，gold 是正确等级的下标。"""
    return Decision(id=id_, source=source, type="score", state=state, instructions=instr,
                    options=[Option(str(k), d) for k, d in enumerate(levels)],
                    target=[1.0 if k == gold else 0.0 for k in range(len(levels))], lang=lang, group=group,
                    meta={"gold": str(gold)})


def table(header: list[str], rows: list[list]) -> str:
    """Markdown 表格。"""
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---- 通用干扰条款：和问题无关的政策段落，用来拉长材料、考验“找到相关规则”的能力 ----------------------
FILLER = {
    "en": [
        "Gift cards are non-refundable and cannot be exchanged for cash except where required by law.",
        "Price adjustments are available within {n} days of purchase if the same item is sold at a lower price by us.",
        "Shipping fees are refunded only when the return is caused by our error, such as a wrong or defective item.",
        "Orders shipped to PO boxes may take up to {n} additional business days.",
        "Loyalty points earned on a purchase are deducted when that purchase is refunded.",
        "Customers may request a copy of any invoice through the account portal for up to {n} months.",
        "Data collected during the transaction is retained for {n} months in accordance with the privacy notice.",
        "International orders may be subject to customs duties, which are the responsibility of the recipient.",
        "Bulk orders of more than {n} units may qualify for a separate commercial agreement.",
        "Promotional codes cannot be combined unless the promotion terms explicitly allow it.",
        "Payment by bank transfer must be received within {n} days, otherwise the order is cancelled automatically.",
        "Customer service is available by chat and email; phone support is limited to business accounts.",
        "Items purchased from third-party sellers on the marketplace follow the seller's own policy.",
        "Disputes not resolved within {n} days may be escalated to the regional consumer office.",
    ],
    "zh": [
        "礼品卡不可退款，也不可兑换现金，法律另有规定的除外。",
        "购买后 {n} 天内，如本店以更低价格出售同一商品，可申请价格补差。",
        "仅当退货原因是我方错误（如发错货或商品有缺陷）时，才退还运费。",
        "寄往邮政信箱的订单可能额外需要 {n} 个工作日。",
        "订单退款时，该订单获得的积分将被扣回。",
        "客户可在账户中心申请开具最近 {n} 个月内的任意发票。",
        "交易过程中收集的数据按照隐私声明保存 {n} 个月。",
        "跨境订单可能产生关税，由收件人承担。",
        "一次采购超过 {n} 件的大宗订单可另行签订商业协议。",
        "优惠码不可叠加使用，除非活动规则明确允许。",
        "银行转账付款须在 {n} 天内到账，否则订单自动取消。",
        "客服通过在线聊天和邮件提供服务，电话支持仅面向企业客户。",
        "平台第三方卖家销售的商品适用卖家自己的售后政策。",
        "{n} 天内未解决的争议可提交给地区消费者保护机构。",
    ],
}


def filler(rng: random.Random, lang: str, k: int) -> list[str]:
    return [s.format(n=rng.choice([3, 5, 7, 10, 14, 30, 60, 90, 180])) for s in rng.sample(FILLER[lang], min(k, len(FILLER[lang])))]


def numbered(clauses: list[str], lang: str) -> str:
    return "\n".join(f"{i}. {c}" for i, c in enumerate(clauses, 1))


# ==== 1. 退款政策 =========================================================================================

CATS = {"en": {"electronics": "Electronics", "apparel": "Apparel", "home": "Home & Kitchen", "toys": "Toys"},
        "zh": {"electronics": "数码电子", "apparel": "服装", "home": "家居厨房", "toys": "玩具"}}


def gen_refund(rng: random.Random, lang: str, long: bool, uid: str) -> tuple[str, list[Decision], str]:
    window = rng.choice([14, 30])
    ext = rng.choice([15, 30])          # 金卡会员延长天数
    fee = rng.choice([10, 15, 20])      # 开封数码商品的重新上架费（%）
    dmg_days = rng.choice([3, 7])       # 破损须在签收后几天内报告
    # 目标订单
    cat = rng.choice(list(CATS["en"]))
    clearance = rng.random() < 0.25
    opened = rng.random() < 0.5
    tier = rng.choice(["standard", "silver", "gold"])
    delivered = D(2026, 1, 1) + dt.timedelta(days=rng.randrange(0, 240))
    days = rng.choice([rng.randrange(1, window + 1), rng.randrange(window - 3, window + ext + 8)])
    days = max(1, days)
    request = delivered + dt.timedelta(days=days)
    damaged = rng.random() < 0.25
    dmg_report = delivered + dt.timedelta(days=rng.randrange(1, 12)) if damaged else None
    price = round(rng.uniform(15, 900), 2)
    # 计算答案（规则按优先级）
    limit = window + (ext if tier == "gold" else 0)
    in_window = days <= limit
    if damaged and (dmg_report - delivered).days <= dmg_days:
        outcome = "full_refund"
    elif clearance:
        outcome = "no_refund"
    elif not in_window:
        outcome = "no_refund"
    elif cat == "electronics" and opened:
        outcome = "partial_refund"
    else:
        outcome = "full_refund"

    order_id = f"A{rng.randrange(100000, 999999)}"
    zh = lang == "zh"
    rules = ([f"标准退货期限为签收后 {window} 天内。",
              f"金卡会员的退货期限在标准期限基础上延长 {ext} 天；银卡会员不享受延长。",
              "标记为“清仓”的商品属于最终销售，不接受退货退款。",
              f"已开封的数码电子商品可以退货，但需扣除商品价格 {fee}% 的重新上架费。",
              f"如果商品在签收时已破损，并在签收后 {dmg_days} 天内报告，无论是否清仓、是否超过退货期限，均可全额退款。"]
             if zh else
             [f"Items may be returned within {window} days of delivery.",
              f"Gold members get an extra {ext} days on top of the standard return window; silver members do not.",
              "Items marked as clearance are final sale and cannot be returned for a refund.",
              f"Opened electronics may be returned, but a restocking fee of {fee}% of the item price is deducted.",
              f"Items that arrive damaged and are reported within {dmg_days} days of delivery receive a full refund, "
              "even if they are clearance items or the return window has passed."])
    clauses = rules + filler(rng, lang, 9 if long else 3)
    rng.shuffle(clauses)

    def row(oid, c, cl, op, dlv, req, dmg, pr):
        return [oid, CATS[lang][c], ("是" if cl else "否") if zh else ("yes" if cl else "no"),
                ("是" if op else "否") if zh else ("yes" if op else "no"), fmt_date(dlv, lang),
                fmt_date(req, lang) if req else "-", fmt_date(dmg, lang) if dmg else "-", f"{pr:.2f}"]

    rows = [row(order_id, cat, clearance, opened, delivered, request, dmg_report, price)]
    for _ in range(rng.randrange(25, 70) if long else rng.randrange(3, 7)):
        d0 = D(2026, 1, 1) + dt.timedelta(days=rng.randrange(0, 240))
        rows.append(row(f"A{rng.randrange(100000, 999999)}", rng.choice(list(CATS["en"])), rng.random() < 0.25,
                        rng.random() < 0.5, d0, None, None, round(rng.uniform(15, 900), 2)))
    rng.shuffle(rows)
    hdr = (["订单号", "品类", "清仓", "已开封", "签收日期", "退货申请日期", "破损报告日期", "价格"] if zh else
           ["order", "category", "clearance", "opened", "delivered", "return requested", "damage reported", "price"])
    tier_txt = {"en": {"standard": "standard", "silver": "silver", "gold": "gold"},
                "zh": {"standard": "普通", "silver": "银卡", "gold": "金卡"}}[lang][tier]
    if zh:
        state = (f"# 退货与退款政策\n{numbered(clauses, lang)}\n\n# 客户信息\n会员等级：{tier_txt}\n\n"
                 f"# 订单记录\n{table(hdr, rows)}\n\n# 本次申请\n客户于 {fmt_date(request, lang)} 申请退回订单 {order_id}。")
        q1 = f"按照政策，订单 {order_id} 的这次退货申请应如何处理？"
        opts = [("full_refund", "全额退款"), ("partial_refund", "扣除重新上架费后部分退款"), ("no_refund", "不予退款")]
        q2 = f"订单 {order_id} 的退货申请在该客户适用的退货期限之内。"
    else:
        state = (f"# Return and refund policy\n{numbered(clauses, lang)}\n\n# Customer\nMembership tier: {tier_txt}\n\n"
                 f"# Order history\n{table(hdr, rows)}\n\n# Current request\nOn {fmt_date(request, lang)} the customer "
                 f"asked to return order {order_id}.")
        q1 = f"Under the policy, how should the return request for order {order_id} be handled?"
        opts = [("full_refund", "Refund the full price"), ("partial_refund", "Refund minus the restocking fee"),
                ("no_refund", "No refund")]
        q2 = f"The return request for order {order_id} falls within the return window that applies to this customer."
    g = f"gen_refund/{uid}"
    return state, [choice(f"{g}/outcome", "gen_refund", state, q1, opts, outcome, lang, g),
                   noul(f"{g}/in_window", "gen_refund", state, q2, in_window, lang, g)], outcome


# ==== 2. 发票对账 =========================================================================================

def gen_invoice(rng: random.Random, lang: str, long: bool, uid: str) -> tuple[str, list[Decision], str]:
    tol = rng.choice([1, 2, 5])
    n = rng.randrange(20, 55) if long else rng.randrange(3, 7)
    skus = rng.sample(range(1000, 9999), n)
    po = [(f"SKU-{s}", rng.randrange(1, 60), round(rng.uniform(2, 400), 2)) for s in skus]
    mode = rng.choice(["ok", "qty", "price", "dup", "price_within"])
    received = {s: q for s, q, _ in po}
    billed = {s: (q, p) for s, q, p in po}
    if mode == "qty":
        s, q, p = rng.choice(po)
        received[s] = max(0, q - rng.randrange(1, max(2, q)))  # 收货少于开票
    elif mode == "price":
        s, q, p = rng.choice(po)
        billed[s] = (q, round(p * (1 + rng.uniform(tol + 0.6, tol + 8) / 100), 2))
    elif mode == "price_within":
        s, q, p = rng.choice(po)
        billed[s] = (q, round(p * (1 + rng.uniform(0.1, tol - 0.4) / 100), 2))
    inv_no = f"INV-{rng.randrange(10000, 99999)}"
    paid = [f"INV-{rng.randrange(10000, 99999)}" for _ in range(rng.randrange(30, 90) if long else rng.randrange(3, 8))]
    if mode == "dup":
        paid.insert(rng.randrange(len(paid) + 1), inv_no)
    # 计算答案（规则按优先级）
    qty_ok = all(billed[s][0] <= received[s] for s, _, _ in po)
    price_ok = all(billed[s][1] <= p * (1 + tol / 100) + 1e-9 for s, _, p in po)
    if inv_no in paid:
        action = "reject_duplicate"
    elif not qty_ok:
        action = "hold_quantity"
    elif not price_ok:
        action = "hold_price"
    else:
        action = "approve"
    zh = lang == "zh"
    rules = ([f"发票号已出现在已付款清单中的，按重复发票拒绝。",
              "任何一行的开票数量超过收货数量的，暂缓付款，等待收货确认。",
              f"任何一行的开票单价高于采购单价超过 {tol}% 的，暂缓付款，等待价格核实；在 {tol}% 以内视为一致。",
              "以上规则按顺序检查，命中第一条即按该条处理；全部通过的发票批准付款。"]
             if zh else
             ["An invoice whose number already appears in the paid-invoice list is rejected as a duplicate.",
              "If any line bills a quantity greater than the quantity received, hold the invoice for receiving confirmation.",
              f"If any line's unit price exceeds the PO unit price by more than {tol}%, hold the invoice for price "
              f"review; differences of {tol}% or less count as matching.",
              "Check the rules in this order and apply the first one that matches; approve invoices that pass all checks."])
    clauses = rules + filler(rng, lang, 6 if long else 2)
    po_rows = [[s, q, f"{p:.2f}"] for s, q, p in po]
    rc_rows = [[s, received[s]] for s, _, _ in po]
    iv_rows = [[s, billed[s][0], f"{billed[s][1]:.2f}"] for s, _, _ in po]
    for r in (rc_rows, iv_rows):
        rng.shuffle(r)
    if zh:
        state = (f"# 应付账款对账规则\n{numbered(clauses, lang)}\n\n# 采购单 PO-{rng.randrange(1000, 9999)}\n"
                 f"{table(['物料', '数量', '单价'], po_rows)}\n\n# 收货单\n{table(['物料', '实收数量'], rc_rows)}\n\n"
                 f"# 发票 {inv_no}\n{table(['物料', '开票数量', '开票单价'], iv_rows)}\n\n# 已付款发票清单\n" + "、".join(paid))
        q1 = f"按照对账规则，发票 {inv_no} 应如何处理？"
        opts = [("approve", "批准付款"), ("hold_quantity", "暂缓：开票数量超过收货数量"),
                ("hold_price", "暂缓：单价差异超过容差"), ("reject_duplicate", "拒绝：重复发票")]
        q2 = f"发票 {inv_no} 每一行的开票数量都不超过收货单上的实收数量。"
    else:
        state = (f"# Accounts payable matching rules\n{numbered(clauses, lang)}\n\n# Purchase order PO-{rng.randrange(1000, 9999)}\n"
                 f"{table(['item', 'qty', 'unit price'], po_rows)}\n\n# Receiving report\n{table(['item', 'qty received'], rc_rows)}\n\n"
                 f"# Invoice {inv_no}\n{table(['item', 'qty billed', 'unit price billed'], iv_rows)}\n\n"
                 "# Paid invoices\n" + ", ".join(paid))
        q1 = f"Under the matching rules, what should happen to invoice {inv_no}?"
        opts = [("approve", "Approve for payment"), ("hold_quantity", "Hold: billed quantity exceeds quantity received"),
                ("hold_price", "Hold: unit price variance above tolerance"), ("reject_duplicate", "Reject as a duplicate invoice")]
        q2 = f"On every line of invoice {inv_no}, the billed quantity is no more than the quantity received."
    g = f"gen_invoice/{uid}"
    return state, [choice(f"{g}/action", "gen_invoice", state, q1, opts, action, lang, g),
                   noul(f"{g}/qty_covered", "gen_invoice", state, q2, qty_ok, lang, g)], action


# ==== 3. 合同期限 =========================================================================================

CONTRACT_FILLER = {
    "en": ["Invoices are payable within {n} days of receipt.", "Either party may assign this agreement only with prior written consent.",
           "Service credits of {n}% apply for each full hour of unplanned downtime beyond the monthly allowance.",
           "Confidential information must be protected for {n} months after the agreement ends.",
           "This agreement is governed by the laws of the State of Delaware.",
           "The customer may audit the provider's security controls once every {n} months.",
           "Fees increase by no more than {n}% at each renewal.", "Notices must be sent to the addresses listed in Schedule B.",
           "Neither party is liable for delays caused by events beyond its reasonable control.",
           "The provider maintains insurance coverage of at least {n} million dollars."],
    "zh": ["发票应在收到后 {n} 天内支付。", "未经对方事先书面同意，任何一方不得转让本协议。",
           "超出每月允许范围的计划外停机，每满一小时按 {n}% 给予服务抵扣。", "协议终止后，保密信息仍须保护 {n} 个月。",
           "本协议适用中华人民共和国法律。", "客户每 {n} 个月可对服务商的安全控制进行一次审计。",
           "每次续约时费用涨幅不超过 {n}%。", "通知应寄送至附件二所列地址。",
           "因超出合理控制范围的事件造成的延误，双方均不承担责任。", "服务商应投保不低于 {n} 百万元的责任险。"],
}


def gen_contract(rng: random.Random, lang: str, long: bool, uid: str) -> tuple[str, list[Decision], str]:
    start = D(2023, 1, 1) + dt.timedelta(days=rng.randrange(0, 700))
    init = rng.choice([12, 24, 36])
    renew = rng.choice([6, 12])
    notice = rng.choice([30, 60, 90])
    amend = None
    if rng.random() < 0.35:  # 修订条款：修订日之后发出的通知适用新的通知期
        amend = (start + dt.timedelta(days=rng.randrange(60, init * 30)), rng.choice([d for d in (30, 60, 90, 120) if d != notice]))
    end0 = add_months(start, init)
    ends = [end0] + [add_months(end0, renew * k) for k in range(1, 12)]
    notice_date = None
    if rng.random() < 0.7:
        notice_date = start + dt.timedelta(days=rng.randrange(30, (init + renew * 2) * 30))
    # 解约生效日：通知日所在期限的期末（若提前量足够），否则下一个期末
    term_effective = None
    if notice_date:
        k_days = amend[1] if amend and notice_date >= amend[0] else notice
        cur = next(e for e in ends if e > notice_date)
        term_effective = cur if (cur - notice_date).days >= k_days else ends[ends.index(cur) + 1]
    query = start + dt.timedelta(days=rng.randrange(30, (init + renew * 3) * 30))
    if term_effective and query >= term_effective:
        status = "terminated"
    elif query < end0:
        status = "initial_term"
    else:
        status = "renewal_term"
    renewed = query >= end0 and (term_effective is None or term_effective > end0)
    zh = lang == "zh"
    f = lambda d: fmt_date(d, lang)
    if zh:
        core = [f"本协议自 {f(start)} 起生效，初始期限为 {init} 个月。",
                f"初始期限届满后，本协议自动续约，每次续约 {renew} 个月，除非一方在当期期限届满前至少 {notice} 天书面通知不再续约。",
                "通知不足上述天数的，本协议在下一个期限届满时终止。"]
        if amend:
            core.append(f"【修订一】自 {f(amend[0])} 起，上述不续约通知期改为 {amend[1]} 天，适用于该日期及之后发出的通知。")
    else:
        core = [f"This agreement takes effect on {f(start)} with an initial term of {init} months.",
                f"After the initial term it renews automatically for successive {renew}-month terms unless either party gives "
                f"written notice of non-renewal at least {notice} days before the end of the then-current term.",
                "A notice given with less than the required period takes effect at the end of the following term."]
        if amend:
            core.append(f"[Amendment 1] For notices given on or after {f(amend[0])}, the non-renewal notice period is "
                        f"{amend[1]} days instead.")
    pool = CONTRACT_FILLER[lang]
    extra = [s.format(n=rng.choice([5, 10, 12, 24, 30, 45])) for s in rng.sample(pool, 10 if long else 3)]
    if long:
        extra += filler(rng, lang, 14)
    clauses = core + extra
    rng.shuffle(clauses)
    log = []
    if notice_date:
        log.append(f"{f(notice_date)}：客户发出书面不续约通知。" if zh else
                   f"{f(notice_date)}: The customer sent written notice of non-renewal.")
    noise = (["双方召开季度业务回顾会。", "服务商发送了年度安全审计报告。", "客户更新了开票联系人。", "双方确认了新的服务等级报告格式。"]
             if zh else ["Quarterly business review held.", "Provider sent the annual security audit report.",
                         "Customer updated its billing contact.", "Both parties agreed a new service report format."])
    for _ in range(rng.randrange(60, 130) if long else rng.randrange(2, 5)):
        log.append(f"{f(start + dt.timedelta(days=rng.randrange(10, 1500)))}：{rng.choice(noise)}" if zh else
                   f"{f(start + dt.timedelta(days=rng.randrange(10, 1500)))}: {rng.choice(noise)}")
    rng.shuffle(log)
    if zh:
        state = f"# 服务协议条款\n{numbered(clauses, lang)}\n\n# 往来记录\n" + "\n".join(log)
        q1 = f"截至 {f(query)}，这份协议处于什么状态？"
        opts = [("initial_term", "仍在初始期限内"), ("renewal_term", "已自动续约，处于续约期内"), ("terminated", "已终止")]
        q2 = f"截至 {f(query)}，这份协议至少已经自动续约过一次。"
    else:
        state = f"# Service agreement terms\n{numbered(clauses, lang)}\n\n# Correspondence log\n" + "\n".join(log)
        q1 = f"As of {f(query)}, what is the status of this agreement?"
        opts = [("initial_term", "Still in the initial term"), ("renewal_term", "Renewed and in a renewal term"),
                ("terminated", "Terminated")]
        q2 = f"By {f(query)}, the agreement has automatically renewed at least once."
    g = f"gen_contract/{uid}"
    return state, [choice(f"{g}/status", "gen_contract", state, q1, opts, status, lang, g),
                   noul(f"{g}/renewed", "gen_contract", state, q2, renewed, lang, g)], status


# ==== 4. 审批路由（多表查找）==============================================================================

LEVELS = ["staff", "manager", "director", "vp"]
APPROVERS = ["manager", "director", "vp", "cfo"]
NAMES_EN = ["Ava", "Ben", "Chen", "Dara", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun", "Kai", "Lena", "Mo", "Nia", "Omar",
            "Pia", "Quinn", "Rui", "Sam", "Tara", "Uma", "Vik", "Wen", "Xia", "Yuri", "Zoe"]
SURN_EN = ["Lee", "Patel", "Kim", "Garcia", "Novak", "Silva", "Okafor", "Wang", "Muller", "Ito", "Rossi", "Haddad"]
NAMES_ZH = list("伟芳娜敏静丽强磊军洋勇艳杰涛明超秀霞平刚桂英华")
SURN_ZH = list("王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗")


def gen_approval(rng: random.Random, lang: str, long: bool, uid: str) -> tuple[str, list[Decision], str]:
    zh = lang == "zh"
    regions = ["APAC", "EMEA", "AMER"]
    depts = rng.sample(["Finance", "Sales", "Legal", "R&D", "Ops", "HR", "Marketing", "IT", "Support", "Procurement"],
                       8 if long else 4)
    dept_region = {d: rng.choice(regions) for d in depts}
    fx = {"USD": 1.0, "EUR": round(rng.uniform(1.02, 1.18), 3), "GBP": round(rng.uniform(1.2, 1.35), 3),
          "JPY": round(rng.uniform(0.0062, 0.0075), 5), "CNY": round(rng.uniform(0.13, 0.145), 4)}
    cats = ["travel", "software", "equipment", "hospitality"]
    th = {c: sorted(rng.sample([500, 1000, 2000, 5000, 10000, 20000, 50000], 3)) for c in cats}
    hosp_region = rng.choice(regions)
    hosp_limit = rng.choice([1000, 2000, 3000])
    people = set()
    while len(people) < (rng.randrange(150, 320) if long else rng.randrange(5, 10)):
        people.add(rng.choice(SURN_ZH) + rng.choice(NAMES_ZH) + rng.choice(NAMES_ZH) if zh else
                   f"{rng.choice(NAMES_EN)} {rng.choice(SURN_EN)}")
    emps = [(p, rng.choice(depts), rng.choices(LEVELS, [5, 3, 2, 1])[0]) for p in sorted(people)]
    rng.shuffle(emps)
    who, dept, level = rng.choice(emps)
    cat = rng.choice(cats)
    cur = rng.choice(list(fx))
    usd = rng.choice([rng.uniform(100, 60000), rng.uniform(th[cat][0] * 0.8, th[cat][-1] * 1.2)])
    amount = round(usd / fx[cur], 0 if cur == "JPY" else 2)
    usd = amount * fx[cur]
    # 计算答案：1) 审批矩阵 2) 地区招待规则 3) 职级规则（审批人至少比申请人高一级）
    base = next((APPROVERS[i] for i, t in enumerate(th[cat]) if usd <= t), "cfo")
    need = APPROVERS.index(base)
    if cat == "hospitality" and dept_region[dept] == hosp_region and usd > hosp_limit:
        need = max(need, APPROVERS.index("vp"))
    if level in ("director", "vp"):
        need = max(need, APPROVERS.index(level) + 1)  # director → 至少 vp；vp → cfo
    approver = APPROVERS[min(need, 3)]
    over = usd > th[cat][1]
    lv = {"zh": {"staff": "员工", "manager": "经理", "director": "总监", "vp": "副总裁"}}.get(lang, {l: l for l in LEVELS})
    ap = {"zh": {"manager": "经理", "director": "总监", "vp": "副总裁", "cfo": "首席财务官"}}.get(lang, {a: a.upper() if a == "cfo" else a for a in APPROVERS})
    cz = {"travel": "差旅", "software": "软件", "equipment": "设备", "hospitality": "招待"}
    mat_rows = [[cz[c] if zh else c, f"≤ {th[c][0]:,}", f"≤ {th[c][1]:,}", f"≤ {th[c][2]:,}", f"> {th[c][2]:,}"] for c in cats]
    hdr_m = ["类别", "经理", "总监", "副总裁", "首席财务官"] if zh else ["category", "manager", "director", "VP", "CFO"]
    emp_rows = [[p, d, lv[l]] for p, d, l in emps]
    dep_rows = [[d, dept_region[d]] for d in depts]
    fx_rows = [[c, r] for c, r in fx.items()]
    if zh:
        rules = ["审批矩阵中的金额均为折算成美元后的金额，按汇率表折算。",
                 f"{hosp_region} 地区部门的招待费用，折算后超过 {hosp_limit:,} 美元的，至少需要副总裁审批。",
                 "申请人职级为总监或副总裁的，审批人必须比申请人至少高一级（总监的申请至少由副总裁审批，副总裁的申请由首席财务官审批）。",
                 "以上规则同时适用，取要求最高的审批级别。"] + filler(rng, lang, 6 if long else 1)
        state = (f"# 费用审批制度\n{numbered(rules, lang)}\n\n# 审批矩阵（美元）\n{table(hdr_m, mat_rows)}\n\n"
                 f"# 汇率（1 单位外币 = ? 美元）\n{table(['币种', '汇率'], fx_rows)}\n\n# 部门\n{table(['部门', '地区'], dep_rows)}\n\n"
                 f"# 员工\n{table(['姓名', '部门', '职级'], emp_rows)}\n\n# 费用申请\n申请人：{who}；类别：{cz[cat]}；金额：{amount:,} {cur}")
        q1 = f"{who} 的这笔费用申请最低需要哪一级审批？"
        opts = [(a, ap[a]) for a in APPROVERS]
        q2 = f"这笔申请折算成美元后超过 {th[cat][1]:,} 美元。"
    else:
        rules = ["All amounts in the approval matrix are in USD after conversion with the FX table.",
                 f"Hospitality expenses from departments in {hosp_region} above {hosp_limit:,} USD need at least VP approval.",
                 "If the requester is a director or VP, the approver must be at least one level above the requester "
                 "(a director's request needs at least a VP; a VP's request needs the CFO).",
                 "All rules apply together; use the highest approval level any rule requires."] + filler(rng, lang, 6 if long else 1)
        state = (f"# Expense approval policy\n{numbered(rules, lang)}\n\n# Approval matrix (USD)\n{table(hdr_m, mat_rows)}\n\n"
                 f"# FX rates (USD per unit)\n{table(['currency', 'rate'], fx_rows)}\n\n# Departments\n{table(['department', 'region'], dep_rows)}\n\n"
                 f"# Employees\n{table(['name', 'department', 'level'], emp_rows)}\n\n"
                 f"# Expense request\nRequester: {who}; category: {cat}; amount: {amount:,} {cur}")
        q1 = f"What is the lowest approval level that {who}'s expense request needs?"
        opts = [(a, ap[a]) for a in APPROVERS]
        q2 = f"Converted to USD, this request is above {th[cat][1]:,} USD."
    g = f"gen_approval/{uid}"
    return state, [choice(f"{g}/approver", "gen_approval", state, q1, opts, approver, lang, g),
                   noul(f"{g}/over_threshold", "gen_approval", state, q2, over, lang, g)], approver


# ==== 5. 工单 SLA（工作时间计算）==========================================================================

def business_hours(a: dt.datetime, b: dt.datetime, holidays: set[D]) -> float:
    """a 到 b 之间的工作时长（小时）：周一到周五 9:00–18:00，节假日除外。"""
    if b <= a:
        return 0.0
    total, day = 0.0, a.date()
    while day <= b.date():
        if day.weekday() < 5 and day not in holidays:
            s = max(a, dt.datetime.combine(day, dt.time(9)))
            e = min(b, dt.datetime.combine(day, dt.time(18)))
            total += max(0.0, (e - s).total_seconds() / 3600)
        day += dt.timedelta(days=1)
    return total


def gen_sla(rng: random.Random, lang: str, long: bool, uid: str) -> tuple[str, list[Decision], str]:
    zh = lang == "zh"
    pri = ["P1", "P2", "P3", "P4"]
    sla = {("standard", p): h for p, h in zip(pri, rng.choice([[2, 4, 8, 16], [1, 4, 9, 18], [2, 6, 12, 24]]))}
    sla.update({("premium", p): max(1, sla[("standard", p)] // 2) for p in pri})
    base = D(2026, 3, 2) + dt.timedelta(days=rng.randrange(0, 200))
    holidays = {base + dt.timedelta(days=rng.randrange(0, 20)) for _ in range(2)}
    customers = [f"{'客户' if zh else 'Cust'}-{rng.randrange(100, 999)}" for _ in range(rng.randrange(30, 80) if long else 5)]
    tiers = {c: rng.choice(["standard", "premium"]) for c in customers}
    tickets = []
    for _ in range(rng.randrange(25, 60) if long else rng.randrange(3, 6)):
        created = dt.datetime.combine(base + dt.timedelta(days=rng.randrange(0, 18)), dt.time(rng.randrange(0, 24), rng.choice([0, 15, 30, 45])))
        reply = created + dt.timedelta(minutes=rng.randrange(20, 60 * 60))
        tickets.append((f"T{rng.randrange(10000, 99999)}", rng.choice(customers), rng.choice(pri), created, reply))
    tid, cust, p, created, reply = rng.choice(tickets)
    limit = sla[(tiers[cust], p)]
    used = business_hours(created, reply, holidays)
    late = used - limit
    status = "met" if late <= 0 else ("breached_minor" if late <= 4 else "breached_major")
    tf = lambda t: (f"{t.month}月{t.day}日 周{WEEKDAY_ZH[t.weekday()]} {t:%H:%M}" if zh else f"{t:%a %Y-%m-%d %H:%M}")
    sla_rows = [[q, sla[("standard", q)], sla[("premium", q)]] for q in pri]
    t_rows = [[t, c, q, tf(a), tf(b)] for t, c, q, a, b in tickets]
    c_rows = [[c, ("高级" if tiers[c] == "premium" else "标准") if zh else tiers[c]] for c in customers]
    hol = "、".join(fmt_date(h, lang) for h in sorted(holidays)) if zh else ", ".join(h.isoformat() for h in sorted(holidays))
    if zh:
        rules = ["首次响应时限按工作时间计算：周一至周五 9:00–18:00，法定节假日不计。工作时间以外创建的工单，从下一个工作时段开始计时。",
                 "时限取决于工单优先级和客户等级，见 SLA 表（单位：工作小时）。", f"本期节假日：{hol}。"] + filler(rng, lang, 5 if long else 1)
        state = (f"# 客服 SLA 规则\n{numbered(rules, lang)}\n\n# SLA 表（首次响应，工作小时）\n{table(['优先级', '标准客户', '高级客户'], sla_rows)}\n\n"
                 f"# 客户等级\n{table(['客户', '等级'], c_rows)}\n\n# 工单\n{table(['工单号', '客户', '优先级', '创建时间', '首次回复时间'], t_rows)}")
        q1 = f"工单 {tid} 的首次响应是否满足 SLA？"
        opts = [("met", "满足时限"), ("breached_minor", "超时，但不超过 4 个工作小时"), ("breached_major", "超时 4 个工作小时以上")]
        q2 = f"工单 {tid} 的首次响应满足 SLA 时限。"
    else:
        rules = ["First-response time is measured in business hours: Monday to Friday, 09:00–18:00, excluding public "
                 "holidays. Tickets created outside business hours start the clock at the next business period.",
                 "The time limit depends on ticket priority and customer tier, as listed in the SLA table (business hours).",
                 f"Public holidays this period: {hol}."] + filler(rng, lang, 5 if long else 1)
        state = (f"# Support SLA rules\n{numbered(rules, lang)}\n\n# SLA table (first response, business hours)\n"
                 f"{table(['priority', 'standard', 'premium'], sla_rows)}\n\n# Customer tiers\n{table(['customer', 'tier'], c_rows)}\n\n"
                 f"# Tickets\n{table(['ticket', 'customer', 'priority', 'created', 'first reply'], t_rows)}")
        q1 = f"Did ticket {tid} meet its first-response SLA?"
        opts = [("met", "Met the time limit"), ("breached_minor", "Breached by at most 4 business hours"),
                ("breached_major", "Breached by more than 4 business hours")]
        q2 = f"Ticket {tid} met its first-response SLA."
    g = f"gen_sla/{uid}"
    return state, [choice(f"{g}/status", "gen_sla", state, q1, opts, status, lang, g),
                   noul(f"{g}/met", "gen_sla", state, q2, late <= 0, lang, g)], status


# ==== 6. 工具调用决策（When2Call 类：该调用工具、追问、直接回答还是说明做不到）=====================

# 工具库：名字 → (英文说明, 中文说明, 必填参数, 请求模板)。模板里 {参数名} 会被填入具体值；
# 去掉某个必填参数的那部分措辞，就得到“信息不全，需要追问”的请求。
TOOLS = {
    "get_weather": ("Get the weather forecast for a city on a date.", "查询某个城市某天的天气预报。", ["city", "date"],
                    {"en": "What will the weather be like in {city} on {date}?", "zh": "{date}{city}的天气怎么样？"}),
    "book_table": ("Reserve a table at a restaurant.", "在餐厅预订座位。", ["restaurant", "time", "party_size"],
                   {"en": "Book a table at {restaurant} for {party_size} people at {time}.",
                    "zh": "帮我在{restaurant}订{time}的位子，{party_size}个人。"}),
    "convert_currency": ("Convert an amount between two currencies.", "把一笔金额从一种货币换算成另一种货币。",
                         ["amount", "from_currency", "to_currency"],
                         {"en": "How much is {amount} {from_currency} in {to_currency}?",
                          "zh": "{amount}{from_currency}能换多少{to_currency}？"}),
    "track_package": ("Track a parcel by its tracking number.", "根据运单号查询包裹物流。", ["tracking_number"],
                      {"en": "Where is my parcel {tracking_number} right now?", "zh": "我的快递{tracking_number}现在到哪了？"}),
    "send_email": ("Send an email.", "发送一封邮件。", ["to", "subject"],
                   {"en": "Email {to} with the subject \"{subject}\".", "zh": "给{to}发封邮件，标题是“{subject}”。"}),
    "search_flights": ("Search flights between two cities on a date.", "查询两地之间某天的航班。",
                       ["origin", "destination", "date"],
                       {"en": "Find flights from {origin} to {destination} on {date}.",
                        "zh": "查一下{date}从{origin}到{destination}的航班。"}),
    "get_stock_price": ("Get the latest price of a stock.", "查询股票的最新价格。", ["ticker"],
                        {"en": "What is {ticker} trading at right now?", "zh": "{ticker}现在股价多少？"}),
    "check_order_status": ("Check the status of an order.", "查询订单状态。", ["order_id"],
                           {"en": "Has my order {order_id} shipped yet?", "zh": "我的订单{order_id}发货了吗？"}),
    "reset_password": ("Send a password reset link to a user account.", "给用户账号发送重置密码链接。", ["username"],
                       {"en": "I forgot my password, my username is {username}.", "zh": "我忘记密码了，用户名是{username}。"}),
    "create_event": ("Create a calendar event.", "创建日历事件。", ["title", "start_time"],
                     {"en": "Put \"{title}\" on my calendar at {start_time}.", "zh": "在日历上加一个“{title}”，时间是{start_time}。"}),
}
VALUES = {
    "city": {"en": ["Paris", "Tokyo", "Berlin", "Toronto"], "zh": ["上海", "杭州", "成都", "巴黎"]},
    "date": {"en": ["Friday", "November 3", "next Monday"], "zh": ["明天", "下周一", "11月3日"]},
    "restaurant": {"en": ["Le Petit Bistro", "Golden Dragon", "Osteria Roma"], "zh": ["外婆家", "鼎泰丰", "海底捞"]},
    "time": {"en": ["7pm", "12:30", "8 tonight"], "zh": ["晚上7点", "中午12点半", "今晚8点"]},
    "party_size": {"en": ["2", "4", "6"], "zh": ["2", "4", "6"]},
    "amount": {"en": ["250", "1,000", "75"], "zh": ["250", "1000", "75"]},
    "from_currency": {"en": ["USD", "EUR", "JPY"], "zh": ["美元", "欧元", "日元"]},
    "to_currency": {"en": ["CNY", "GBP", "CAD"], "zh": ["人民币", "英镑", "港币"]},
    "tracking_number": {"en": ["SF1234567890", "1Z999AA10123456784"], "zh": ["SF1234567890", "YT8899001122"]},
    "to": {"en": ["alice@example.com", "the finance team"], "zh": ["财务部", "王经理"]},
    "subject": {"en": ["Q3 budget", "Meeting moved"], "zh": ["三季度预算", "会议改期"]},
    "origin": {"en": ["Boston", "Madrid", "Seattle"], "zh": ["北京", "深圳", "西安"]},
    "destination": {"en": ["Chicago", "Lisbon", "Denver"], "zh": ["成都", "厦门", "昆明"]},
    "ticker": {"en": ["AAPL", "TSLA", "NVDA"], "zh": ["AAPL", "TSLA", "NVDA"]},
    "order_id": {"en": ["#58213", "A-99120"], "zh": ["#58213", "A-99120"]},
    "username": {"en": ["jdoe", "maria.k"], "zh": ["zhangsan", "lily_w"]},
    "title": {"en": ["Dentist", "Team sync"], "zh": ["看牙医", "团队周会"]},
    "start_time": {"en": ["Friday 3pm", "9am tomorrow"], "zh": ["周五下午3点", "明早9点"]},
}
# 缺失参数在请求里换成的含糊说法（按参数类型，读起来要自然）
VAGUE = {
    "en": {"city": "a city", "origin": "my city", "destination": "somewhere", "date": "some day", "time": "later",
           "start_time": "sometime", "restaurant": "a restaurant", "party_size": "a few", "amount": "some money",
           "from_currency": "my currency", "to_currency": "another currency", "tracking_number": "it",
           "to": "someone", "subject": "something", "ticker": "that company", "order_id": "it",
           "username": "my account", "title": "an appointment"},
    "zh": {"city": "那边", "origin": "这里", "destination": "那边", "date": "改天", "time": "晚点",
           "start_time": "某个时间", "restaurant": "一家餐厅", "party_size": "几", "amount": "一些",
           "from_currency": "这边的钱", "to_currency": "另一种货币", "tracking_number": "那个", "to": "某人",
           "subject": "某件事", "ticker": "那家公司", "order_id": "那个", "username": "我的账号", "title": "一个安排"},
}
# 不需要工具、助手自己就能回答的请求，以及需要外部能力、但手头没有合适工具的请求。
DIRECT = {"en": ["What is 15% of 80?", "Rewrite this more politely: send me the file now.",
                 "What does the acronym API stand for?", "Give me a synonym for 'quick'."],
          "zh": ["80 的 15% 是多少？", "把这句话改得礼貌一点：马上把文件发给我。", "API 这个缩写是什么意思？", "“迅速”有什么近义词？"]}
CANNOT = {"en": ["Book me a hotel room in Rome for next weekend.", "Order a large pizza to my office.",
                 "Transfer $500 from my savings to my checking account.", "Turn off the lights in my living room."],
          "zh": ["帮我订下周末罗马的酒店。", "给我办公室点一份大号披萨。", "从储蓄账户转 500 元到活期账户。", "把我客厅的灯关掉。"]}
TOOL_ACTIONS = ["call_tool", "ask_for_info", "answer_directly", "cannot_help"]
TOOL_ACTION_DESC = {
    "en": {"call_tool": "Call one of the available tools; the request contains everything that tool needs.",
           "ask_for_info": "Ask the user for missing information that a suitable tool requires.",
           "answer_directly": "Answer from its own knowledge without calling any tool.",
           "cannot_help": "Explain that it cannot help: no available tool fits and it cannot do this itself."},
    "zh": {"call_tool": "调用某个可用工具；请求里已经包含该工具需要的全部信息。",
           "ask_for_info": "向用户追问合适的工具所需、但请求里缺少的信息。",
           "answer_directly": "不调用工具，直接凭自身知识回答。",
           "cannot_help": "说明无法完成：没有合适的工具，自己也做不到。"},
}


def gen_tool(rng: random.Random, lang: str, long: bool, uid: str) -> tuple[str, list[Decision], str]:
    zh = lang == "zh"
    action = rng.choice(TOOL_ACTIONS)
    names = list(TOOLS)
    if action in ("call_tool", "ask_for_info"):
        target = rng.choice(names)
        others = [n for n in names if n != target]
        shown = [target] + rng.sample(others, rng.randrange(2, len(others) + 1))
        desc_en, desc_zh, req, tmpl = TOOLS[target]
        vals = {k: rng.choice(VALUES[k][lang]) for k in req}
        if action == "ask_for_info":  # 去掉一个必填参数：把它的值换成含糊的说法（助手必须追问）
            miss = rng.choice(req)
            vals[miss] = VAGUE[lang][miss]
        text = tmpl[lang].format(**vals)
        gold_tool = target
    else:
        shown = rng.sample(names, rng.randrange(3, len(names) + 1))
        text = rng.choice(DIRECT[lang] if action == "answer_directly" else CANNOT[lang])
        gold_tool = "none"
    # 长材料：再加 20–40 个无关的“伪工具”，工具选择题因此常常超过 26 个选项（练习两字母编码）
    fake = []
    if long:
        for k in range(rng.randrange(20, 41)):
            fake.append((f"internal_api_{rng.randrange(1000, 9999)}_{k}",
                         rng.choice(["Read an internal log file.", "Rotate service credentials.", "Export a CRM report.",
                                     "Recompute warehouse stock levels.", "Archive old chat transcripts."])))
    tools = [{"name": n, "description": TOOLS[n][1] if zh else TOOLS[n][0], "required": TOOLS[n][2]} for n in shown]
    tools += [{"name": n, "description": d, "required": []} for n, d in fake]
    rng.shuffle(tools)
    state = state_to_text({"tools": tools, "user": text})
    g = f"gen_tool/{uid}"
    q1 = "助手应该如何回应这条请求？" if zh else "How should the assistant respond to this request?"
    opts1 = [(a, TOOL_ACTION_DESC[lang][a]) for a in TOOL_ACTIONS]
    tool_opts = [(t["name"], t["description"]) for t in tools] + [("none", "不需要调用任何工具" if zh else "No tool is needed or suitable")]
    q2 = "处理这条请求应该使用哪个工具？" if zh else "Which tool is relevant to this request?"
    return state, [choice(f"{g}/action", "gen_tool", state, q1, opts1, action, lang, g),
                   choice(f"{g}/tool", "gen_tool", state, q2, tool_opts, gold_tool, lang, g)], action


FAMILIES: dict[str, Callable] = {"refund": gen_refund, "invoice": gen_invoice, "contract": gen_contract,
                                 "approval": gen_approval, "sla": gen_sla}
# ==== 7. 长对话客服工单（读完整段多轮对话：归哪个队列、要不要升级、客户情绪）==========================

# 客服工单的 10 个团队队列（与 ajev/data/more_sources.py 的 TICKET_QUEUES 相同，这里复制一份避免循环导入）
TICKET_QUEUES_FOR_GEN = {
    "Technical Support": "Technical problems with the product or service: errors, bugs, configuration.",
    "Product Support": "Questions about using product features or how a product works.",
    "Customer Service": "General customer requests, complaints and account matters.",
    "IT Support": "Internal IT issues: hardware, network, accounts and access.",
    "Billing and Payments": "Invoices, charges, refunds and payment methods.",
    "Returns and Exchanges": "Returning or exchanging purchased items.",
    "Service Outages and Maintenance": "Service downtime, outages and planned maintenance.",
    "Sales and Pre-Sales": "Pricing, quotes, demos and questions before buying.",
    "General Inquiry": "Requests that do not fit any specific team.",
    "Human Resources": "Employment, payroll, benefits and other HR matters.",
}
CHAT_ISSUES = {  # 问题类型 → (队列, 英文开场, 中文开场)
    "delivery": ("Customer Service", "My order still hasn't arrived.", "我的订单到现在还没到。"),
    "double_charge": ("Billing and Payments", "I was charged twice for the same order.", "同一个订单被扣了两次钱。"),
    "damaged": ("Returns and Exchanges", "The blender I received is cracked.", "收到的搅拌机外壳裂了。"),
    "lockout": ("IT Support", "I can't log in, my account says it's locked.", "我登不上账号，提示账号被锁定了。"),
    "outage": ("Service Outages and Maintenance", "Your app has been down all morning.", "你们的应用一上午都打不开。"),
    "how_to": ("Product Support", "How do I export my data to CSV?", "怎么把数据导出成 CSV？"),
}
ANGRY = {"en": ["This is ridiculous.", "I'm really fed up with this.", "Why does this keep happening?!",
                "This is the worst service I've had.", "I've wasted hours on this."],
         "zh": ["这也太离谱了。", "我真的受够了。", "怎么老是出这种问题？！", "这是我遇到过最差的服务。", "我在这上面浪费了好几个小时。"]}
NEUTRAL_C = {"en": ["Sure, one moment.", "My email is the one on the account.", "OK.", "Yes, that's right.",
                    "I'm checking now.", "Thanks."],
             "zh": ["好的，稍等。", "邮箱就是账号上那个。", "嗯。", "对，没错。", "我看一下。", "谢谢。"]}
NEUTRAL_A = {"en": ["Thanks for reaching out. Can you confirm your email?", "Let me look into that for you.",
                    "Could you share your order number?", "I'm checking with the team now.",
                    "Thanks for your patience.", "Is there anything else I can help with?"],
             "zh": ["感谢联系我们，能确认一下邮箱吗？", "我帮您查一下。", "方便提供订单号吗？", "我正在和团队确认。",
                    "感谢您的耐心等待。", "还有其他可以帮您的吗？"]}
LEGAL = {"en": "If this isn't fixed today I'm filing a chargeback with my bank.", "zh": "今天再不解决我就向银行申请拒付。"}


def gen_chat(rng: random.Random, lang: str, long: bool, uid: str) -> tuple[str, list[Decision], str]:
    zh = lang == "zh"
    issue = rng.choice(list(CHAT_ISSUES))
    queue, open_en, open_zh = CHAT_ISSUES[issue]
    vip = rng.random() < 0.3
    contacts = rng.choice([1, 1, 2, 3, 4])           # 这是第几次就同一问题联系
    hours = rng.choice([4, 12, 30, 60, 90])          # 问题已经持续多少小时
    legal = rng.random() < 0.2
    n_angry = rng.choice([0, 0, 1, 2, 3, 4])
    turns = [("customer", open_zh if zh else open_en)]
    n_fill = rng.randrange(70, 140) if long else rng.randrange(3, 8)
    for _ in range(n_fill):
        turns.append(("agent", rng.choice(NEUTRAL_A[lang])))
        turns.append(("customer", rng.choice(NEUTRAL_C[lang])))
    # 生气的话、法律威胁插在对话后半段（必须读到后面才能发现）
    late = len(turns) // 2
    for _ in range(n_angry):
        turns.insert(rng.randrange(late, len(turns) + 1), ("customer", rng.choice(ANGRY[lang])))
    if legal:
        turns.insert(rng.randrange(late, len(turns) + 1), ("customer", LEGAL[lang]))
    escalate = legal or contacts >= 3 or (vip and hours > 48)
    anger = min(3, n_angry + (1 if legal else 0))
    who = {"customer": "客户" if zh else "Customer", "agent": "客服" if zh else "Agent"}
    convo = "\n".join(f"{who[r]}: {t}" for r, t in turns)
    if zh:
        meta = f"客户等级：{'VIP' if vip else '普通'}；本问题已持续 {hours} 小时；这是客户第 {contacts} 次就此问题联系我们。"
        rules = ("升级规则：客户提出拒付或法律行动；或同一问题第 3 次及以上联系；或 VIP 客户的问题持续超过 48 小时——"
                 "满足任意一条就升级给主管。")
        state = f"# 工单信息\n{meta}\n{rules}\n\n# 对话记录\n{convo}"
        q1, q2, q3 = "这张工单应该分到哪个队列？", "按照升级规则，这张工单需要升级给主管。", "到对话结束时，客户的情绪激动程度是？"
        levels = ["平静", "有些不满", "明显生气", "非常愤怒"]
    else:
        meta = (f"Customer tier: {'VIP' if vip else 'standard'}; issue open for {hours} hours; "
                f"this is contact #{contacts} about this issue.")
        rules = ("Escalation rule: escalate to a supervisor if the customer threatens a chargeback or legal action, "
                 "or this is the 3rd or later contact about the same issue, or a VIP customer's issue has been open "
                 "for more than 48 hours.")
        state = f"# Ticket\n{meta}\n{rules}\n\n# Conversation\n{convo}"
        q1, q2, q3 = ("Which queue should this ticket go to?", "Under the escalation rule, this ticket must be escalated.",
                      "By the end of the conversation, how upset is the customer?")
        levels = ["calm", "somewhat annoyed", "clearly angry", "furious"]
    g = f"gen_chat/{uid}"
    opts = [(q, d) for q, d in TICKET_QUEUES_FOR_GEN.items()]
    return state, [choice(f"{g}/queue", "gen_chat", state, q1, opts, queue, lang, g),
                   noul(f"{g}/escalate", "gen_chat", state, q2, escalate, lang, g),
                   score(f"{g}/anger", "gen_chat", state, q3, levels, anger, lang, g)], str(escalate)


# ==== 8. 安全日志（从大量登录日志里判断账号是否被盗、严重程度）====================================

COUNTRIES = ["US", "DE", "BR", "IN", "NG", "RU", "VN", "CN", "GB", "FR"]


def gen_seclog(rng: random.Random, lang: str, long: bool, uid: str) -> tuple[str, list[Decision], str]:
    zh = lang == "zh"
    users = [f"user{rng.randrange(100, 999)}" for _ in range(rng.randrange(10, 20) if long else 4)]
    target = rng.choice(users)
    home = {u: rng.choice(COUNTRIES[:5]) for u in users}
    t0 = dt.datetime(2026, 9, rng.randrange(1, 28), 0, 0)
    events = []
    for u in users:  # 正常活动：在本国的成功登录
        for _ in range(rng.randrange(2, 6) if long else rng.randrange(1, 3)):
            events.append((t0 + dt.timedelta(minutes=rng.randrange(0, 1440)), u, "login_success", home[u]))
    # 目标账号的场景
    pattern = rng.choice(["benign", "failed_only", "brute_then_success", "impossible_travel", "priv_esc"])
    base = t0 + dt.timedelta(minutes=rng.randrange(60, 1200))
    foreign = rng.choice([c for c in COUNTRIES if c != home[target]])
    if pattern in ("failed_only", "brute_then_success", "priv_esc"):
        for k in range(rng.randrange(5, 12)):
            events.append((base + dt.timedelta(seconds=20 * k), target, "login_failed", foreign))
    if pattern in ("brute_then_success", "priv_esc"):
        events.append((base + dt.timedelta(minutes=5), target, "login_success", foreign))
    if pattern == "priv_esc":
        events.append((base + dt.timedelta(minutes=9), target, "role_changed_to_admin", foreign))
    if pattern == "impossible_travel":
        events.append((base, target, "login_success", home[target]))
        events.append((base + dt.timedelta(minutes=rng.randrange(10, 50)), target, "login_success", foreign))
    events.sort()
    compromised = pattern in ("brute_then_success", "impossible_travel", "priv_esc")
    severity = {"benign": 0, "failed_only": 1, "brute_then_success": 2, "impossible_travel": 2, "priv_esc": 3}[pattern]
    log = "\n".join(f"{t:%Y-%m-%d %H:%M:%S} user={u} event={e} country={c}" for t, u, e, c in events)
    if zh:
        rules = ("判定规则：同一账号多次登录失败后、从异常国家登录成功，或 1 小时内在两个国家登录成功（不可能的移动），"
                 "视为账号可能被盗；被盗后又获得管理员权限为最严重。仅有登录失败而没有成功登录为低风险。"
                 f"各账号常用国家见账号表。")
        acct = "\n".join(f"{u}: 常用国家 {home[u]}" for u in users)
        state = f"# 规则\n{rules}\n\n# 账号表\n{acct}\n\n# 认证日志\n{log}"
        q1, q2 = f"账号 {target} 很可能已经被盗用。", f"账号 {target} 相关事件的严重程度是？"
        levels = ["无异常", "低", "中", "高"]
    else:
        rules = ("Rules: a successful login from an unusual country after repeated failed logins, or successful logins "
                 "from two countries within one hour (impossible travel), means the account is likely compromised; "
                 "gaining admin rights after that is the most severe case. Failed logins with no success are low risk. "
                 "Each account's usual country is listed in the account table.")
        acct = "\n".join(f"{u}: usual country {home[u]}" for u in users)
        state = f"# Rules\n{rules}\n\n# Accounts\n{acct}\n\n# Authentication log\n{log}"
        q1, q2 = f"Account {target} is likely compromised.", f"How severe are the events for account {target}?"
        levels = ["none", "low", "medium", "high"]
    g = f"gen_seclog/{uid}"
    return state, [noul(f"{g}/compromised", "gen_seclog", state, q1, compromised, lang, g),
                   score(f"{g}/severity", "gen_seclog", state, q2, levels, severity, lang, g)], pattern


# ==== 9. HR 休假政策（政策 + 员工档案 + 申请 → 批准 / 需经理审批 / 拒绝）==============================

def gen_hr(rng: random.Random, lang: str, long: bool, uid: str) -> tuple[str, list[Decision], str]:
    zh = lang == "zh"
    notice_req = rng.choice([7, 14])
    probation = 90
    kind = rng.choice(["annual", "sick", "parental"])
    tenure = rng.choice([30, 60, 120, 200, 400, 800])
    days = rng.randrange(1, 12)
    balance = rng.randrange(0, 20)
    notice = rng.randrange(0, 30)
    cert = rng.random() < 0.5
    if kind == "annual":
        if tenure < probation:
            out = "reject"
        elif balance < days:
            out = "reject"
        elif notice < notice_req:
            out = "needs_manager"
        else:
            out = "approve"
    elif kind == "sick":
        out = "approve" if days <= 3 or cert else "reject"
    else:
        out = "approve" if tenure >= 365 else "reject"
    direct = out == "approve"
    names = [rng.choice(SURN_ZH) + rng.choice(NAMES_ZH) if zh else f"{rng.choice(NAMES_EN)} {rng.choice(SURN_EN)}"
             for _ in range(rng.randrange(40, 120) if long else 5)]
    who = names[0]
    rows = []
    for k, n in enumerate(names):
        rows.append([n, tenure if k == 0 else rng.choice([30, 120, 400, 900]), balance if k == 0 else rng.randrange(0, 20)])
    rng.shuffle(rows)
    kz = {"annual": "年假", "sick": "病假", "parental": "育儿假"}
    if zh:
        rules = [f"入职未满 {probation} 天（试用期）的员工不能休年假。", "年假天数不能超过剩余年假余额。",
                 f"年假需至少提前 {notice_req} 天申请，否则需要经理特批。", "病假超过 3 天须提供医院证明，否则不予批准。",
                 "育儿假要求入职满 365 天。"] + filler(rng, lang, 6 if long else 1)
        state = (f"# 休假制度\n{numbered(rules, lang)}\n\n# 员工档案\n{table(['姓名', '入职天数', '剩余年假'], rows)}\n\n"
                 f"# 申请\n{who} 申请{kz[kind]} {days} 天，提前 {notice} 天提交"
                 + ("，附有医院证明。" if kind == "sick" and cert else "。"))
        q1, q2 = f"{who} 的这份休假申请应该如何处理？", f"按照制度，{who} 的这份申请不需要任何人特批，可以直接批准。"
        opts = [("approve", "直接批准"), ("needs_manager", "需要经理特批"), ("reject", "不予批准")]
    else:
        rules = [f"Employees within their first {probation} days (probation) cannot take annual leave.",
                 "Annual leave cannot exceed the remaining annual leave balance.",
                 f"Annual leave must be requested at least {notice_req} days ahead, otherwise it needs manager approval.",
                 "Sick leave longer than 3 days requires a doctor's note, otherwise it is not approved.",
                 "Parental leave requires at least 365 days of service."] + filler(rng, lang, 6 if long else 1)
        state = (f"# Leave policy\n{numbered(rules, lang)}\n\n# Employee records\n"
                 f"{table(['name', 'days of service', 'annual leave balance'], rows)}\n\n# Request\n{who} requests "
                 f"{days} days of {kind} leave, submitted {notice} days in advance"
                 + (", with a doctor's note." if kind == "sick" and cert else "."))
        q1, q2 = f"How should {who}'s leave request be handled?", f"Under the policy, {who}'s request can be approved directly without anyone's special approval."
        opts = [("approve", "Approve"), ("needs_manager", "Needs manager approval"), ("reject", "Reject")]
    g = f"gen_hr/{uid}"
    return state, [choice(f"{g}/decision", "gen_hr", state, q1, opts, out, lang, g),
                   noul(f"{g}/direct", "gen_hr", state, q2, direct, lang, g)], out


# ==== 10. 急诊分诊（按分诊规则判断紧急程度；只判断就诊优先级，不做诊断）==============================

def gen_triage(rng: random.Random, lang: str, long: bool, uid: str) -> tuple[str, list[Decision], str]:
    zh = lang == "zh"
    pats = []
    for _ in range(rng.randrange(20, 60) if long else 4):
        pats.append({"id": f"P{rng.randrange(1000, 9999)}", "age": rng.choice([1, 8, 30, 45, 62, 80]),
                     "spo2": rng.choice([86, 92, 95, 98]), "sbp": rng.choice([82, 105, 125, 160]),
                     "temp": rng.choice([36.8, 38.2, 39.4]), "pain": rng.randrange(0, 11),
                     "chest_pain": rng.random() < 0.15, "confused": rng.random() < 0.1})
    p = rng.choice(pats)

    def level(p) -> int:  # 1 = 立即，2 = 紧急，3 = 次紧急，4 = 非紧急
        if p["spo2"] < 90 or p["sbp"] < 90 or p["confused"] or (p["chest_pain"] and p["age"] >= 40):
            return 1
        if (p["temp"] >= 39 and (p["age"] <= 1 or p["age"] >= 75)) or p["pain"] >= 8:
            return 2
        if p["temp"] >= 38 or p["pain"] >= 4:
            return 3
        return 4
    lv = level(p)
    yn = (lambda b: "是" if b else "否") if zh else (lambda b: "yes" if b else "no")
    rows = [[x["id"], x["age"], x["spo2"], x["sbp"], x["temp"], x["pain"], yn(x["chest_pain"]), yn(x["confused"])] for x in pats]
    if zh:
        rules = ["一级（立即处置）：血氧低于 90%，或收缩压低于 90，或意识模糊，或 40 岁及以上伴胸痛。",
                 "二级（紧急）：体温 ≥39℃ 且年龄 ≤1 岁或 ≥75 岁，或疼痛评分 ≥8。",
                 "三级（次紧急）：体温 ≥38℃，或疼痛评分 ≥4。", "四级（非紧急）：其余情况。",
                 "从一级开始逐级判断，满足即停止。本规则只用于确定就诊顺序，不是诊断。"] + filler(rng, lang, 4 if long else 0)
        hdr = ["编号", "年龄", "血氧%", "收缩压", "体温", "疼痛评分", "胸痛", "意识模糊"]
        state = f"# 分诊规则\n{numbered(rules, lang)}\n\n# 候诊患者\n{table(hdr, rows)}"
        q1, q2 = f"患者 {p['id']} 的分诊级别是？", f"患者 {p['id']} 需要立即处置（一级）。"
        levels = ["四级（非紧急）", "三级（次紧急）", "二级（紧急）", "一级（立即处置）"]
    else:
        rules = ["Level 1 (immediate): SpO2 below 90%, or systolic BP below 90, or confusion, or chest pain at age 40 or older.",
                 "Level 2 (emergent): temperature ≥39°C with age ≤1 or ≥75, or pain score ≥8.",
                 "Level 3 (urgent): temperature ≥38°C, or pain score ≥4.", "Level 4 (non-urgent): everything else.",
                 "Check from level 1 downward and stop at the first match. These rules set the order of care; "
                 "they are not a diagnosis."] + filler(rng, lang, 4 if long else 0)
        hdr = ["id", "age", "SpO2 %", "systolic BP", "temp °C", "pain score", "chest pain", "confused"]
        state = f"# Triage rules\n{numbered(rules, lang)}\n\n# Waiting patients\n{table(hdr, rows)}"
        q1, q2 = f"What is patient {p['id']}'s triage level?", f"Patient {p['id']} needs immediate care (level 1)."
        levels = ["Level 4 (non-urgent)", "Level 3 (urgent)", "Level 2 (emergent)", "Level 1 (immediate)"]
    g = f"gen_triage/{uid}"
    return state, [score(f"{g}/level", "gen_triage", state, q1, levels, 4 - lv, lang, g),
                   noul(f"{g}/immediate", "gen_triage", state, q2, lv == 1, lang, g)], str(lv)


EXTRA_FAMILIES: dict[str, Callable] = {"tool": gen_tool, "chat": gen_chat, "seclog": gen_seclog, "hr": gen_hr,
                                       "triage": gen_triage}  # 不在默认列表里，保证 lm2 的生成结果可复现


def generate(n_decisions: int, seed: int, zh_share: float = 0.35, long_share: float = 0.3,
             split: str = "t", families: list[str] | None = None) -> list[Decision]:
    """生成约 n_decisions 道题（每份材料 2 道），五类平均分配；每类按主问题的答案均衡（拒绝采样）。"""
    rng = random.Random(seed)
    fams = {f: {**FAMILIES, **EXTRA_FAMILIES}[f] for f in families} if families else FAMILIES
    per_family = n_decisions // 2 // len(fams)
    out: list[Decision] = []
    seen: set[int] = set()
    for fam, fn in fams.items():
        counts: Counter = Counter()
        n_labels = {"refund": 3, "invoice": 4, "contract": 3, "approval": 4, "sla": 3, "tool": 4,
                    "chat": 2, "seclog": 5, "hr": 3, "triage": 4}[fam]
        cap = per_family // n_labels + 1
        made, tries = 0, 0
        while made < per_family and tries < per_family * 200:
            tries += 1
            lang = "zh" if rng.random() < zh_share else "en"
            long = rng.random() < long_share
            state, ds, label = fn(rng, lang, long, f"{split}{made:05d}")
            if counts[label] >= cap or hash(state) in seen:
                continue
            for d in ds:
                d.meta["long"] = long
                d.validate()
            seen.add(hash(state))
            counts[label] += 1
            made += 1
            out.extend(ds)
        print(f"[hard_gen] {fam}: {made} states, labels {dict(counts)}")
    rng.shuffle(out)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4500, help="number of decisions (2 per state)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split", default="t", help="id prefix, e.g. t for train and d for dev")
    ap.add_argument("--zh-share", type=float, default=0.35)
    ap.add_argument("--long-share", type=float, default=0.3)
    a = ap.parse_args(argv)
    ds = generate(a.n, a.seed, a.zh_share, a.long_share, a.split)
    write_jsonl(a.out, ds)
    print(f"[hard_gen] {len(ds)} decisions -> {a.out}")


if __name__ == "__main__":
    main()
