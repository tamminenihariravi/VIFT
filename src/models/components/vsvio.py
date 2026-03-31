import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0.0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )


class Inertial_encoder(nn.Module):
    def __init__(self, opt):
        super(Inertial_encoder, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout))

        self.proj = nn.Linear(256 * 1 * 11, opt.i_f_len)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))
        x = self.encoder_conv(x.permute(0, 2, 1))
        out = self.proj(x.view(x.shape[0], -1))
        return out.view(batch_size, seq_len, 256)


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        self.conv1 = conv(True, 6, 64, kernel_size=7, stride=2, dropout=0.2)
        self.conv2 = conv(True, 64, 128, kernel_size=5, stride=2, dropout=0.2)
        self.conv3 = conv(True, 128, 256, kernel_size=5, stride=2, dropout=0.2)
        self.conv3_1 = conv(True, 256, 256, kernel_size=3, stride=1, dropout=0.2)
        self.conv4 = conv(True, 256, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv4_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv5 = conv(True, 512, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv5_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv6 = conv(True, 512, 1024, kernel_size=3, stride=2, dropout=0.5)

        __tmp = Variable(torch.zeros(1, 6, opt.img_w, opt.img_h))
        __tmp = self.encode_image(__tmp)

        self.visual_head = nn.Linear(int(np.prod(__tmp.size())), opt.v_f_len)
        self.inertial_encoder = Inertial_encoder(opt)

    def forward(self, img, imu):
        v = torch.cat((img[:, :-1], img[:, 1:]), dim=2)
        batch_size = v.size(0)
        seq_len = v.size(1)

        v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4))
        v = self.encode_image(v)
        v = v.view(batch_size, seq_len, -1)
        v = self.visual_head(v)

        imu = torch.cat([imu[:, i * 10:i * 10 + 11, :].unsqueeze(1) for i in range(seq_len)], dim=1)
        imu = self.inertial_encoder(imu)
        return v, imu

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6
# -------------------------------------------------------------------
# బ్యాచ్ నార్మలైజేషన్ (Batch Normalization) అంటే ఏమిటి?
# -------------------------------------------------------------------
# ఒక డీప్ న్యూరల్ నెట్వర్క్లో చాలా లేయర్స్ ఉంటాయి.
# మనం డేటాను మొత్తం ఒకేసారి ఇవ్వకుండా చిన్న చిన్న భాగాలుగా (Mini-batches) ఇస్తాము.
#
# ❌ సమస్య (Internal Covariate Shift):
# ట్రైనింగ్ జరుగుతున్నప్పుడు, ఒక బ్యాచ్ ఇచ్చినప్పుడు 3వ లేయర్ ఔట్పుట్ (Distribution)
# ఒకలా ఉంటుంది, మరో బ్యాచ్ ఇచ్చినప్పుడు ఆ ఔట్పుట్ పూర్తిగా మారిపోతుంది.
# ఇలా ప్రతిసారి డేటా పంపిణీ (Distribution) మారిపోతూ ఉంటే, ఆ తర్వాతి లేయర్ (4వ లేయర్)
# ఆ మార్పులకు తగ్గట్టు తనను తాను మార్చుకోవడానికి చాలా కష్టపడుతుంది.
# దీనివల్ల మోడల్ ట్రైనింగ్ చాలా నెమ్మదిగా మరియు కష్టంగా మారుతుంది.
#
# ✅ పరిష్కారం (Batch Normalization):
# నెట్వర్క్ లోపల ప్రతి లేయర్ దగ్గర మనం ఆ చిన్న బ్యాచ్ డేటాకు
# సగటు (Mean) మరియు వేరియన్స్ (Variance) కనుక్కుని, దాన్ని నార్మలైజ్ చేస్తారు.
# అంటే ప్రతి బ్యాచ్ డేటాను సగటు '0' కి, వేరియన్స్ '1' కి తీసుకొస్తారు.
# దీనివల్ల తర్వాత లేయర్కి వెళ్లే డేటా ఎప్పుడూ ఒకే స్టాండర్డ్ ఫార్మాట్లో ఉంటుంది.
# అప్పుడు నెట్వర్క్ చాలా ఈజీగా నేర్చుకుంటుంది.
#
#   x̂ = (x - μ) / √(σ² + ε)    ← నార్మలైజేషన్ (ε చిన్న సంఖ్య, zero divide నివారణ)
#
# 📝 గమనిక (Train vs Test):
# ట్రైనింగ్ అప్పుడు ఆయా బ్యాచ్ల సగటును వాడతారు.
# కానీ మోడల్ ట్రైనింగ్ పూర్తయ్యాక, టెస్టింగ్ (Validation) సమయంలో మాత్రం
# మొత్తం డేటాసెట్ యొక్క సగటు/వేరియన్స్ ను (Running Statistics) వాడతారు.
#
# 🎛️ γ (Scale) మరియు β (Shift) పారామీటర్లు:
# మనం డేటాను బలవంతంగా '0' సగటు, '1' వేరియన్స్ కి మార్చేస్తున్నాం.
# కానీ కొన్నిసార్లు డేటా అలా ఉండకపోవడమే మోడల్ నేర్చుకోవడానికి మంచిది కావొచ్చు!
# అందుకే నార్మలైజ్ చేసిన తర్వాత, మోడల్ కి స్వేచ్ఛని ఇవ్వడానికి
# γ (స్కేల్) మరియు β (షిఫ్ట్) అనే రెండు learnable పారామీటర్లను కలుపుతారు.
#
#   y = γ * x̂ + β    ← మోడల్ స్వయంగా నేర్చుకునే స్కేల్ మరియు షిఫ్ట్
#
# దీని అర్థం ఏమిటంటే: మోడల్ కనుక "నాకు ఈ నార్మలైజ్ చేసిన డేటా వద్దు,
# పాత డేటానే కావాలి" అని భావిస్తే, మోడల్ స్వయంగా γ, β లను అడ్జస్ట్ చేసుకుని
# తిరిగి పాత డేటాను తెచ్చుకోగలదు.
# అంటే, మనం డేటాను ఒక స్టాండర్డ్ ప్లేస్ కి తీసుకొచ్చి,
# "ఇక్కడి నుండి నీకు ఏది మంచిదో అది నేర్చుకో" అని మోడల్కి వదిలేస్తాం!
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# nn.Linear(in_features, out_features) అనేది ఒక linear transformation చేస్తుంది:
#
# 👉 గణితంగా:
#   y = Wx + b
#   x → input vector
#   W → weights (matrix)
#   b → bias
#   y → output
#
# 🔷 Parameters వివరణ
#
# 1. in_features
#    👉 input size (input vector length)
#    ఉదాహరణ: input = [x1, x2, x3, x4] అయితే → in_features = 4
#
# 2. out_features
#    👉 output neurons సంఖ్య
#    ఉదాహరణ: out_features = 2 అంటే → output [y1, y2]
#
# 3. bias (optional)
#    👉 default: True
#    bias=True  → y = Wx + b
#    bias=False → y = Wx
#
# 🔷 Internal Structure
#
#    nn.Linear(256*11, opt.i_f_len) అయితే:
#      Weight matrix size = (opt.i_f_len × 2816)
#      Bias size          = (opt.i_f_len)
#
#    అంటే:
#      W = [ [w11 w12 ... w1,2816]   ← row for output neuron 1
#            [w21 w22 ... w2,2816]   ← row for output neuron 2
#             ...                                              ]
#      b = [b1, b2, ..., b_i_f_len]
# -------------------------------------------------------------------


# ─────────────────────────────────────────────────────────────────
# Inertial_encoder.forward()
# ─────────────────────────────────────────────────────────────────
# x: (N, seq_len, 11, 6)
#
# What does '11' mean?
# ─────────────────────────────────────────────────────────────────
# '11' represents the number of IMU readings inside ONE frame interval
# (i.e., between two consecutive camera frames).
#
#   Camera frame rate = 10 Hz  →  1 frame every 0.1 sec
#   IMU rate          = 100 Hz →  100 readings per sec
#
#   IMU readings per frame interval = 100 / 10 = 10 gaps
#                                   → 10 gaps + 1 endpoint = 11 readings
#
#   So: 11 = IMU measurements between two consecutive camera frames
#                                                                   ┌─ frame 1
#   Timeline:  |──●──●──●──●──●──●──●──●──●──●──●──|
#              0  1  2  3  4  5  6  7  8  9  10  ← 11 IMU samples
#                                                     └─ frame 2

# What does 'seq_len' mean?
# ─────────────────────────────────────────────────────────────────
# seq_len = how many frame-pairs (windows) we process together in one sample.
#
#   Frame1 → Frame2   (pair 1)
#   Frame2 → Frame3   (pair 2)
#   Frame3 → Frame4   (pair 3)
#   Frame4 → Frame5   (pair 4)
#
#   If seq_len = 4, we process 4 consecutive frame pairs at once.
#   Each pair has 11 IMU samples → total IMU shape: (N, 4, 11, 6)
# ─────────────────────────────────────────────────────────────────

# x: (N x seq_len, 11, 6)

# x.permute(0, 2, 1) swaps the last two dimensions:
#   Before permute: x → (N × seq_len,  11,  6)
#                         │             │    └── 6 IMU channels (ax,ay,az,gx,gy,gz)
#                         │             └─────── 11 time steps
#                         └───────────────────── batch
#
#   After  permute: x → (N × seq_len,   6,  11)
#                                        │    └── 11 time steps (now the "length" for Conv1d)
#                                        └─────── 6 channels (Conv1d expects: batch, channels, length)
#
# Why? Because nn.Conv1d expects input shape: (batch, in_channels, length)
# So the 6 IMU axes must be in the channels dimension, and 11 time steps in the length dimension.

# x: (N x seq_len, 64, 11)

# x.view(x.shape[0], -1) keeps the batch dimension and flattens all others:
#   x is currently: (N × seq_len,  256,  11)
#                    │             │     └── 11 time steps (length)
#                    │             └──────── 256 channels (after Conv1d)
#                    └────────────────────── batch (B)
#
#   x.view(x.shape[0], -1)  →  (N × seq_len,  256 × 11)
#                               │              └── flattened: C × L = 2816
#                               └── batch dimension kept as-is
#
#   This flattened vector is then fed into self.proj (a Linear layer)
#   which maps 2816 → opt.i_f_len (the final IMU feature size).

# out: (N x seq_len, 256)


# ─────────────────────────────────────────────────────────────────
# class Encoder
# ─────────────────────────────────────────────────────────────────
#
# 1. __init__ ఫంక్షన్ (మోడల్ నిర్మాణం / Architecture)
# ─────────────────────────────────────────────────────────────────
#
# 🔷 CNN లేయర్స్ (self.conv1 నుండి self.conv6 వరకు):
# ─────────────────────────────────────────────────────────────────
# ఇక్కడ మనం ఇందాక మాట్లాడుకున్న conv అనే హెల్పర్ ఫంక్షన్ని వాడారు.
# (ప్రతి బ్లాక్లో Conv2d + BatchNorm + LeakyReLU + Dropout ఉంటాయి).
#
# ఒక ముఖ్యమైన లాజిక్: మొదటి లేయర్ conv1 లో ఇన్పుట్ ఛానెల్స్ 6 అని ఇచ్చారు.
# సాధారణంగా కలర్ ఇమేజ్కి 3 ఛానెల్స్ (RGB) ఉంటాయి. ఇక్కడ 6 ఎందుకు ఉన్నాయంటే...
# ప్రస్తుత ఫ్రేమ్ను, తర్వాతి ఫ్రేమ్ను (ఉదా: t1, t2) కలిపి (Stack చేసి) మోడల్కి
# ఇస్తున్నారు (Optical Flow మోడల్స్ లాగా). అందుకే 3+3 = 6 ఛానెల్స్ అయ్యాయి.
#
# ఈ లేయర్స్ గుండా వెళ్ళేకొద్దీ ఇమేజ్ సైజు సగానికి తగ్గుతూ (stride=2),
# ఛానెల్స్ సంఖ్య (6 ➔ 64 ➔ 128 ➔ 512 ➔ 1024) పెరుగుతూ వస్తుంది.
#
# 🔷 డమ్మీ టెన్సార్ ట్రిక్ (Dummy Tensor Trick):
# ─────────────────────────────────────────────────────────────────
#   __tmp = Variable(torch.zeros(1, 6, opt.img_w, opt.img_h))
#   __tmp = self.encode_image(__tmp)
#   self.visual_head = nn.Linear(int(np.prod(__tmp.size())), opt.v_f_len)
#
# CNN లేయర్స్ అంతా అయిపోయాక చివర్లో అవుట్పుట్ సైజు (1024 x H x W) ఎంత వస్తుందో
# మ్యాన్యువల్ గా లెక్కపెట్టడం కష్టం. అందుకే ఒక డమ్మీ ఇమేజ్ను (అన్నీ సున్నాలు ఉన్న
# టెన్సార్) క్రియేట్ చేసి, దాన్ని నెట్వర్క్ గుండా పంపించి, చివరగా వచ్చిన సైజును బట్టి
# ఆటోమెటిక్గా ఒక Linear లేయర్ను క్రియేట్ చేస్తున్నారు.
#
# చివరగా IMU డేటాను ప్రాసెస్ చేయడానికి self.inertial_encoder ని రెడీ చేశారు.
# ─────────────────────────────────────────────────────────────────
#
# 2. forward ఫంక్షన్ (డేటా ప్రయాణించే దారి)
# ─────────────────────────────────────────────────────────────────
#
# 🅰️ ఇమేజ్ (Visual) డేటా ప్రాసెసింగ్:
# v = torch.cat((img[:, :-1], img[:, 1:]), dim=2):
#   ప్రస్తుత ఇమేజ్ మరియు తర్వాతి ఇమేజ్ లను కలుపుతున్నారు (6 ఛానెల్స్ అవుతాయి).
#   దీనివల్ల మోడల్ ఆ రెండు ఇమేజ్ల మధ్య ఉన్న "కదలికను (Motion)" అర్థం చేసుకుంటుంది.
#
# v.view(batch_size * seq_len, ...):
#   సీక్వెన్స్లన్నింటినీ Flatten చేసి CNN లోకి పంపిస్తున్నారు.
#
# self.encode_image(v) → CNN లేయర్స్ (conv1 టు conv6) గుండా వెళ్ళి వస్తుంది.
#
# v.view(batch_size, seq_len, -1) → తిరిగి (Batch, Sequence) ఆకారంలోకి మారుస్తున్నారు.
#
# self.visual_head → Linear లేయర్కి పంపి opt.v_f_len సైజు ఫీచర్లు రాబడుతున్నారు.
#
# 🅱️ సెన్సార్ (IMU) డేటా ప్రాసెసింగ్:
# కెమెరా ఒక్క ఫోటో తీసే లోపు, IMU సెన్సార్ 10 సార్లు రీడింగ్స్ తీసుకుంటుంది.
# కాబట్టి ప్రతి 10 సెన్సార్ రీడింగ్స్ ని (+1 అదనంగా) ఒక గ్రూపుగా కట్ చేస్తున్నారు.
# కట్ చేసిన ఆ సెన్సార్ డేటాను self.inertial_encoder(imu) కి పంపి ఫీచర్లను తీస్తున్నారు.
#
# చివరగా విజువల్ ఫీచర్లను (v) మరియు IMU ఫీచర్లను (imu) రిటర్న్ చేస్తున్నారు.
#
# 3. encode_image ఫంక్షన్
# ─────────────────────────────────────────────────────────────────
# CNN బ్లాక్స్ అన్నింటినీ ఒకదాని తర్వాత ఒకటి వరుసగా కలుపుతుంది.
# (conv1 ➔ conv2 ➔ ... ➔ conv6)
#
# 📌 సారాంశం (Summary)
# ─────────────────────────────────────────────────────────────────
# ఈ క్లాస్ అనేది ఒక "సెన్సార్ ఫ్యూజన్ ఎన్కోడర్" (Sensor Fusion Encoder).
# రెండు వరుస కెమెరా ఫోటోలు (Vision) + సెన్సార్ రీడింగ్స్ (Inertial) →
# వాటిని తర్వాతి Transformer లేయర్స్కి పంపే 768-dim features గా మారుస్తుంది.
# ─────────────────────────────────────────────────────────────────
