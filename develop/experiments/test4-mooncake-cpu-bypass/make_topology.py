#!/usr/bin/env python3
"""生成 h200-2 CPU/GPU/NIC/PCIe-switch 拓扑图 (SVG),并转为 JPEG。
数据源:h200-2 实测 `nvidia-smi topo -m` / lspci / numactl(2026-07-22)。
用法: python3 make_topology.py  ->  topology_h200-2.svg + topology_h200-2.jpg
"""
import cairosvg
from PIL import Image
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1600, 1010
# cairosvg 用 cairo toy font,不做逐字形 fallback,须用单一含 CJK+Latin 的字体。
# a100-2 /root/.fonts/MergedCJKLatin.ttf 正是合并字体(中英文全覆盖)。
FONT = "MergedCJKLatin"

# 每 GPU 一条 lane;(GPU 名, NIC 名, NUMA)
NUMA0 = [("GPU0", "CX-7\nmlx5_0"), ("GPU1", "CX-7\nmlx5_1"),
         ("GPU2", "CX-7\nmlx5_2"), ("GPU3", "CX-7\nmlx5_3")]
NUMA1 = [("GPU4", "CX-7\nmlx5_4"), ("GPU5", "CX-7\nmlx5_5"),
         ("GPU6", "CX-7\nmlx5_6"), ("GPU7", "CX-7\nmlx5_9")]

C = dict(bg="#0f172a", panel="#1e293b", gpu="#16a34a", gpu_s="#065f16",
         nvsw="#7c3aed", pcie="#2563eb", nic="#ea580c", bond="#dc2626",
         cpu="#334155", dram="#64748b", txt="#f1f5f9", sub="#cbd5e1",
         line="#94a3b8", nvlink="#a78bfa")

LANE_W, GAP = 150, 25
x0 = 80                      # NUMA0 起点
x1 = 850                     # NUMA1 起点
def lane_x(group_start, i):
    return group_start + i * (LANE_W + GAP)

s = []
def rect(x, y, w, h, fill, rx=8, stroke="none", sw=1, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
def line(x1_, y1_, x2_, y2_, col=C["line"], sw=2, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    s.append(f'<line x1="{x1_}" y1="{y1_}" x2="{x2_}" y2="{y2_}" '
             f'stroke="{col}" stroke-width="{sw}"{d}/>')
def text(x, y, t, size=15, col=C["txt"], anchor="middle", weight="normal"):
    for k, ln in enumerate(t.split("\n")):
        s.append(f'<text x="{x}" y="{y + k*(size+2)}" font-size="{size}" '
                 f'fill="{col}" text-anchor="{anchor}" font-weight="{weight}" '
                 f'font-family="{FONT}">{ln}</text>')

s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">')
rect(0, 0, W, H, C["bg"], rx=0)
text(W/2, 38, "h200-2 硬件拓扑:CPU / GPU / NIC / PCIe Switch", 26, C["txt"], weight="bold")
text(W/2, 62, "2×Xeon Platinum 8558 (2 NUMA) · 8×H200 (NVSwitch 全互联) · 8×ConnectX-7 (每GPU一张,GPUDirect) · PEX890xx Gen5 switch",
     14, C["sub"])

# NVSwitch 顶栏
nvsw_y = 82
rect(70, nvsw_y, W-140, 44, C["nvsw"], rx=10)
text(W/2, nvsw_y+28, "NVSwitch fabric — 8×GPU 全互联 NVLink (NV18 ≈ 18-link bonded, ~900 GB/s/GPU)",
     16, "#ffffff", weight="bold")

GPU_Y, GPU_H = 178, 66
BOX_Y, BOX_H = 162, 300      # 虚线 PCIe switch domain
NIC_Y, NIC_H = 356, 62
CPU_Y, CPU_H = 540, 92

def draw_lane(gx, gpu, nic):
    # PCIe switch domain(虚线框)
    rect(gx, BOX_Y, LANE_W, BOX_H, "none", rx=10, stroke=C["pcie"], sw=2, dash="7 5")
    text(gx+LANE_W/2, BOX_Y+BOX_H-10, "PCIe Switch", 12, C["pcie"])
    text(gx+LANE_W/2, BOX_Y+BOX_H+6, "(PEX890xx)", 11, C["pcie"])
    # GPU
    rect(gx+10, GPU_Y, LANE_W-20, GPU_H, C["gpu"], stroke=C["gpu_s"], sw=2)
    text(gx+LANE_W/2, GPU_Y+30, gpu, 18, "#ffffff", weight="bold")
    text(gx+LANE_W/2, GPU_Y+50, "H200 141GB", 11, "#dcfce7")
    # GPU -> NVSwitch
    line(gx+LANE_W/2, GPU_Y, gx+LANE_W/2, nvsw_y+44, C["nvlink"], 3)
    # NIC (CX-7)
    rect(gx+18, NIC_Y, LANE_W-36, NIC_H, C["nic"], stroke="#9a3412", sw=2)
    text(gx+LANE_W/2, NIC_Y+26, nic.split("\n")[0], 13, "#ffffff", weight="bold")
    text(gx+LANE_W/2, NIC_Y+44, nic.split("\n")[1], 12, "#ffedd5")
    # GPU—NIC 同一 switch = PIX
    line(gx+LANE_W/2, GPU_Y+GPU_H, gx+LANE_W/2, NIC_Y, C["line"], 2)
    text(gx+LANE_W/2+4, (GPU_Y+GPU_H+NIC_Y)/2+4, "PIX", 11, "#fbbf24", anchor="start")

for i, (g, n) in enumerate(NUMA0):
    draw_lane(lane_x(x0, i), g, n)
for i, (g, n) in enumerate(NUMA1):
    draw_lane(lane_x(x1, i), g, n)

# 每个 switch domain -> CPU
n0_l, n0_r = x0, lane_x(x0, 3) + LANE_W
n1_l, n1_r = x1, lane_x(x1, 3) + LANE_W
for i in range(4):
    gxl = lane_x(x0, i) + LANE_W/2
    line(gxl, BOX_Y+BOX_H, gxl, CPU_Y, C["line"], 2)
    gxr = lane_x(x1, i) + LANE_W/2
    line(gxr, BOX_Y+BOX_H, gxr, CPU_Y, C["line"], 2)

# CPU sockets
rect(n0_l, CPU_Y, n0_r-n0_l, CPU_H, C["cpu"], stroke="#0f172a", sw=2)
text((n0_l+n0_r)/2, CPU_Y+38, "CPU0 · Xeon Platinum 8558 (48c)", 17, "#ffffff", weight="bold")
text((n0_l+n0_r)/2, CPU_Y+62, "NUMA node0 · CPU 0-47 · PCIe Gen5 Root", 13, C["sub"])
rect(n1_l, CPU_Y, n1_r-n1_l, CPU_H, C["cpu"], stroke="#0f172a", sw=2)
text((n1_l+n1_r)/2, CPU_Y+38, "CPU1 · Xeon Platinum 8558 (48c)", 17, "#ffffff", weight="bold")
text((n1_l+n1_r)/2, CPU_Y+62, "NUMA node1 · CPU 48-95 · PCIe Gen5 Root", 13, C["sub"])

# UPI
line(n0_r, CPU_Y+CPU_H/2, n1_l, CPU_Y+CPU_H/2, "#f59e0b", 4)
text((n0_r+n1_l)/2, CPU_Y+CPU_H/2-10, "UPI", 15, "#fbbf24", weight="bold")
text((n0_r+n1_l)/2, CPU_Y+CPU_H/2+22, "(跨NUMA=SYS)", 11, "#fbbf24")

# DRAM
DR_Y = CPU_Y+CPU_H+22
rect(n0_l, DR_Y, 330, 46, C["dram"], rx=8)
text(n0_l+165, DR_Y+29, "DDR5  ~1 TB (NUMA0)", 14, "#ffffff")
line(n0_l+165, CPU_Y+CPU_H, n0_l+165, DR_Y, C["line"], 2)
rect(n1_r-330, DR_Y, 330, 46, C["dram"], rx=8)
text(n1_r-165, DR_Y+29, "DDR5  ~1 TB (NUMA1)", 14, "#ffffff")
line(n1_r-165, CPU_Y+CPU_H, n1_r-165, DR_Y, C["line"], 2)

# mlx5_bond_0 (CX-6 Dx bond) 挂 NUMA1
bx, bw = n1_l, 360
rect(bx, DR_Y, bw, 46, C["bond"], rx=8)
text(bx+bw/2, DR_Y+20, "ConnectX-6 Dx 双口 bond = mlx5_bond_0", 13, "#ffffff", weight="bold")
text(bx+bw/2, DR_Y+38, "bond1 10.119.195.74(前端/管理网,非GPUDirect)", 11, "#fecaca")
line(bx+bw/2, CPU_Y+CPU_H, bx+bw/2, DR_Y, C["line"], 2, dash="4 3")

# 图例
LG_Y = DR_Y+72
rect(70, LG_Y, W-140, 168, C["panel"], rx=10, stroke="#334155", sw=1)
text(90, LG_Y+26, "图例 / 连接层级(源自 nvidia-smi topo -m)", 15, C["txt"], anchor="start", weight="bold")
legends = [
    ("NV18", C["nvlink"], "GPU↔GPU 18-link NVLink(经 NVSwitch,全 8 卡 all-to-all)"),
    ("PIX", "#fbbf24", "GPU 与其 NIC 同一 PCIe switch(单桥)→ GPUDirect RDMA 最短路径"),
    ("NODE", C["line"], "同 NUMA 内跨 PCIe host bridge(如 GPU0↔mlx5_1)"),
    ("SYS", "#f59e0b", "跨 NUMA,经 UPI(如 GPU0↔GPU4 的 PCIe 路径 / mlx5_bond_0↔GPU0-3)"),
]
yy = LG_Y+50
for tag, col, desc in legends:
    s.append(f'<rect x="92" y="{yy-13}" width="20" height="16" rx="3" fill="{col}"/>')
    text(120, yy, tag, 13, C["txt"], anchor="start", weight="bold")
    text(180, yy, desc, 13, C["sub"], anchor="start")
    yy += 27
text(90, yy+4,
     "对 PD 实验的意义:P(GPU0-3)↔D(GPU4-7)的 KV 传输——nixl 用 UCX cuda_ipc 走 NVLink(NV18,免网卡);",
     13, "#a7f3d0", anchor="start")
text(90, yy+24,
     "mooncake TransferEngine 仅 rdma/tcp,须经 ConnectX NIC(单机回环)且需 RoCEv2 GID,故单机 PD 不占 NVLink 优势。",
     13, "#a7f3d0", anchor="start")

s.append('</svg>')
svg = "\n".join(s)

svg_path = os.path.join(HERE, "topology_h200-2.svg")
jpg_path = os.path.join(HERE, "topology_h200-2.jpg")
with open(svg_path, "w") as f:
    f.write(svg)

png_bytes = cairosvg.svg2png(bytestring=svg.encode(), output_width=1600, output_height=1010)
img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
img.save(jpg_path, "JPEG", quality=92)
print("wrote", svg_path)
print("wrote", jpg_path, img.size)
