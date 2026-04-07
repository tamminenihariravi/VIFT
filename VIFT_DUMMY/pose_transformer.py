from torch import nn
import torch
import math


# ఈ కోడ్‌లో మొత్తం 4 వేర్వేరు క్లాసులు (Models) ఉన్నాయి. ఎందుకు 4 రాశారో, ఒక్కోటి ఏం చేస్తుందో సులభంగా చూద్దాం:
# 1. PoseTransformer (అసలైన హీరో)
# ఇది కెమెరా (Visual - 512) మరియు సెన్సార్ (Inertial - 256) రెండింటినీ కలిపి (మొత్తం 768 సైజులో ఉన్న డేటాను) తీసుకునే ప్రధాన మోడల్.
# దీనిలో 3 ముఖ్యమైన ఫీచర్స్ ఉన్నాయి:
# Positional Embedding (సమయ గుర్తింపు): మీరు మోడల్‌కి 11 ఫ్రేమ్స్ ఇస్తే, ఏది మొదటి ఫ్రేమ్, ఏది రెండో ఫ్రేమ్ అని మోడల్‌కి తెలియదు (ట్రాన్స్‌ఫార్మర్లకి కళ్లు ఉండవు కదా). అందుకే ప్రతి ఫ్రేమ్ కి ఒక సీరియల్ నంబర్ (టైమ్ స్టాంప్) ని అతికించే టెక్నిక్ ఇది.
# Causal Mask (భవిష్యత్తు చూడకుండా మాస్క్): 5వ సెకనులో కారు ఎక్కడుందో అంచనా వేయాలంటే, మోడల్ కేవలం 1, 2, 3, 4 సెకన్ల డేటాను మాత్రమే చూడాలి. పొరపాటున 6వ సెకను డేటాను (భవిష్యత్తును) చూడకూడదు. దానికోసం కళ్లకు గంతలు కట్టే పద్ధతే ఈ generate_square_subsequent_mask.
# Output (ఫలితం): ఇది నేరుగా కారు ఎంత కదిలిందో చెప్పే 6 నంబర్లను (Regression) అవుట్‌పుట్‌గా ఇస్తుంది.
# 2. TokenizedPoseTransformer (అత్యంత అడ్వాన్స్‌డ్ & ఆసక్తికరమైన మోడల్)
# ఇది చాలా వినూత్నమైన ఐడియా! మామూలుగా కారు కదలికను చెప్పాలంటే "1.5 మీటర్లు ముందుకి" అని నంబర్లలో చెబుతారు.
# కానీ ఈ మోడల్‌లో నంబర్లను పదాలుగా (Tokens గా) మార్చేశారు!
# అంటే ChatGPT అక్షరాలను పసిగట్టి "తర్వాతి పదం ఏంటి?" అని అంచనా వేసినట్లే... ఈ మోడల్ పాత కదలికలను చూసి "తర్వాతి కదలిక (Token) ఏంటి?" అని ఊహిస్తుంది.
# దీనికోసం ప్రత్యేకంగా ఒక tokenizer (పదకోశం) ని క్రియేట్ చేశారు.
# గుర్తుందా? మనం మొదట్లో మాట్లాడుకున్న "డీప్ ఇంబ్యాలెన్స్‌డ్ రిగ్రెషన్" సమస్యను పరిష్కరించడానికి ఇది ఒక అద్భుతమైన మార్గం. నంబర్లను అంచనా వేయడం కష్టం, కానీ ఇలా టోకెన్స్ లాగా (Classification లాగా) మార్చేస్తే.. చాలా అరుదుగా వచ్చే టర్నింగ్స్ (Rare cases) ని కూడా ఈ మోడల్ ఈజీగా గుర్తుపడుతుంది!
# 3. PoseTransformerVisual (కేవలం కళ్లు మాత్రమే)
# ఈ మోడల్ పేరులోనే ఉంది. డెవలపర్ ఈ లైన్ చూడండి:
# visual_inertial_features[:,:,:512]
# మొత్తం 768 డేటాలో, కేవలం మొదటి 512 (అంటే కెమెరా ఇమేజెస్ డేటాను) మాత్రమే కట్ చేసి తీసుకుంటున్నారు. సెన్సార్ (IMU) డేటాను పూర్తిగా తీసేశారు!
# లక్ష్యం: "సెన్సార్ పాడైపోయి, కేవలం ఇమేజెస్ మాత్రమే ఉంటే నా మోడల్ కారు కదలికను ఎంత కచ్చితంగా అంచనా వేయగలదు?" అని చెక్ చేయడానికి రాసిన మోడల్ ఇది.
# 4. PoseTransformerInertial (కేవలం సెన్సార్ మాత్రమే)
# ఇది పైదానికి రివర్స్.
# visual_inertial_features[:,:,512:]
# మొత్తం డేటాలో కెమెరా ఇమేజెస్ (512) ని వదిలేసి.. కేవలం చివరి 256 (అంటే సెన్సార్ / IMU డేటాను) మాత్రమే తీసుకుంటున్నారు.
# లక్ష్యం: "రాత్రి పూట కెమెరాకి ఏమీ కనిపించకపోతే, కేవలం బ్యాలెన్స్ సెన్సార్లతో మోడల్ కారు కదలికను అంచనా వేయగలదా?" అని టెస్ట్ చేయడానికి రాసిన మోడల్.
# 💡 అసలు ఒక డెవలపర్ ఒకే కోడ్‌లో 4 మోడల్స్ ఎందుకు రాస్తాడు? (Ablation Study)
# రీసెర్చ్ పేపర్స్ రాసేటప్పుడు ఎవరైనా ఒకే మోడల్ రాసి "ఇదే బెస్ట్" అని చెబితే ఎవరూ నమ్మరు. అందుకే రీసెర్చర్స్ ఇలా రకరకాల మోడల్స్ రాస్తారు:
# "చూడండి.. కేవలం ఇమేజ్ ఇస్తే (Visual) మోడల్ 70% కరెక్ట్ ఆన్సర్ చెప్పింది."
# "కేవలం సెన్సార్ ఇస్తే (Inertial) 60% చెప్పింది."
# "రెండూ కలిపి ఇస్తే (PoseTransformer) 90% చెప్పింది!"
# "రెండూ కలిపి టోకెన్స్ వాడి ఇస్తే (TokenizedPoseTransformer) 95% ఆక్యూరసీ వచ్చింది!"
# ఇలా ప్రూవ్ చేసి తమ టెక్నాలజీ (టోకనైజేషన్ + సెన్సార్ ఫ్యూజన్) ఎంత గ్రేట్ అని నిరూపించుకోవడానికి (దీన్నే రీసెర్చ్ భాషలో Ablation Study అంటారు) ఈ కోడ్ మొత్తం డిజైన్ చేశారు!




# 🧠 అటెన్షన్ ఫార్ములా (Attention Formula) — ట్రాన్స్‌ఫార్మర్ గుండెకాయ
# ════════════════════════════════════════════════════════════════════════════════
#
# ట్రాన్స్ఫార్మర్ (Transformer) మోడల్స్ (ChatGPT లాంటివి) సృష్టించిన అద్భుతాలన్నింటికీ
# గుండెకాయ లాంటిది ఈ అటెన్షన్ ఫార్ములా (Attention Formula).
#
# రీసెర్చ్ పేపర్ "Attention Is All You Need" లో ఇచ్చిన ఆ ఫార్ములా ఇదిగో:
#
#   Attention(Q, K, V) = softmax( (Q × K^T) / √d_k ) × V
#
# చూడటానికి మ్యాథ్స్ లాగా భయపెట్టేలా ఉన్నా, దీని వెనుక ఉన్న లాజిక్ చాలా చాలా సింపుల్!
# దీన్ని ఒక "యూట్యూబ్ సెర్చ్ (YouTube Search)" ఉదాహరణతో వివరిస్తాను.
#
# ──────────────────────────────────────────────────────────────────────────────
# 1. ఆ మూడు అక్షరాలు ఏంటి? (Q, K, V)
# ──────────────────────────────────────────────────────────────────────────────
#
# Q (Query - మీ ప్రశ్న):
#   మీరు యూట్యూబ్ సెర్చ్ బార్లో టైప్ చేసేది (ఉదా: "Python tutorials in Telugu").
#   అంటే "మీకు ఏం కావాలి?" అని అడిగేది Q.
#
# K (Key - వీడియో టైటిల్స్):
#   యూట్యూబ్లో ఉన్న కోట్ల వీడియోలకు ఉన్న టైటిల్స్ లేదా ట్యాగ్స్.
#   అంటే "నా దగ్గర ఈ సమాచారం ఉంది" అని చెప్పేవి K.
#
# V (Value - అసలు వీడియో):
#   ఆ టైటిల్ (Key) వెనుక ఉన్న అసలైన కంటెంట్/వీడియోనే V.
#   అంటే "ఫైనల్ గా మీకు దొరికే ఆన్సర్".
#
# ──────────────────────────────────────────────────────────────────────────────
# 2. ఫార్ములా ఎలా పనిచేస్తుంది? (Step-by-Step)
# ──────────────────────────────────────────────────────────────────────────────
#
# స్టెప్ 1: మ్యాచ్లను వెతకడం (Q × K^T)
#   మ్యాథ్స్లో రెండు వెక్టార్లను గుణించడం (Dot Product) అంటే..
#   ఆ రెండూ ఎంత "సిమిలర్ (సారూప్యత)" గా ఉన్నాయో కనుక్కోవడం.
#   ఇక్కడ మీ క్వెరీని (Q), డేటాబేస్ లో ఉన్న ప్రతి కీ (K) తో గుణిస్తారు.
#   (K^T అంటే ట్రాన్స్పోజ్ - గుణించడానికి వీలుగా మ్యాట్రిక్స్ను తిప్పడం).
#   మీ ప్రశ్న (Q) కి ఏ వీడియో టైటిల్ (K) ఎక్కువ కనెక్ట్ అవుతుందో,
#   దానికి పెద్ద నంబర్ (స్కోర్) వస్తుంది. సంబంధం లేని వాటికి చిన్న నంబర్ లేదా మైనస్ వస్తుంది.
#   (ఉదాహరణ స్కోర్లు: వీడియో1=100, వీడియో2=10, వీడియో3=-50).
#
# స్టెప్ 2: కూల్ చేయడం / బ్యాలెన్స్ చేయడం (÷ √d_k)
#   పైన వచ్చిన స్కోర్లు ఒక్కోసారి మరీ పెద్దగా (ఉదా: 1000, 5000) అయిపోతాయి.
#   నంబర్లు మరీ పెద్దవైతే మోడల్ బ్రెయిన్ హ్యాంగ్ అవుతుంది (Gradient Vanishing అంటారు).
#   అందుకే ఆ స్కోర్లను కంట్రోల్ చేయడానికి, డేటా సైజు యొక్క స్క్వేర్ రూట్ (√d_k)
#   తో భాగిస్తారు (Scaling). ఇది స్కోర్లను నార్మల్గా ఉంచుతుంది.
#
# స్టెప్ 3: శాతాలుగా మార్చడం (softmax)
#   స్కోర్లు వచ్చేశాయి కదా, ఇప్పుడు ఆ స్కోర్లను శాతాలుగా
#   (Percentages: 0 నుండి 1 మధ్య) మార్చాలి. దానికి వాడే ఫంక్షనే softmax.
#   ఇది ఏం చేస్తుందంటే.. ఎక్కువ స్కోర్ వచ్చిన దానికి ఎక్కువ శాతం,
#   తక్కువ దానికి తక్కువ శాతం ఇస్తుంది (అన్నీ కలిపితే 100% లేదా 1 అవ్వాలి).
#   (ఉదాహరణ: వీడియో1 = 0.85 (85%), వీడియో2 = 0.12 (12%), వీడియో3 = 0.03 (3%)).
#   ఈ శాతాలనే "Attention Weights (అటెన్షన్ వెయిట్స్)" అంటారు.
#   అంటే మోడల్ ఏ వీడియో మీద ఎంత దృష్టి (Attention) పెట్టాలి అని నిర్ణయించే నంబర్లు.
#
# స్టెప్ 4: అసలైన ఆన్సర్ని తీసుకోవడం (× V)
#   ఇప్పుడు మన దగ్గర శాతాలు (Weights) ఉన్నాయి. అసలైన కంటెంట్ (V) ఉంది.
#   మోడల్ ఇప్పుడు ఆ శాతాలను వాడి అసలైన వీడియోలను కలుపుతుంది!
#   అంటే: (0.85 × వీడియో1) + (0.12 × వీడియో2) + (0.03 × వీడియో3).
#   మనకు కావాల్సిన సమాచారం వీడియో1 లో 85% ఉంది కాబట్టి,
#   ఫైనల్ ఆన్సర్లో వీడియో1 కంటెంట్ ఎక్కువగా కనిపిస్తుంది.
#
# ════════════════════════════════════════════════════════════════════════════════
# 👇 కింద ఉన్న TransformerEncoder లోపల ఈ అటెన్షన్ ఫార్ములా పనిచేస్తుంది!
# ════════════════════════════════════════════════════════════════════════════════

class PoseTransformer(nn.Module):
    def __init__(self, input_dim=768, embedding_dim=128, num_layers=2, nhead=8, dim_feedforward=512, dropout=0.1):
        super(PoseTransformer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
        )
#         ఇన్‌పుట్ (input_dim = 768): మనం పాత కోడ్‌తో జ్యూస్ (Latent Vector) తీసి .npy ఫైల్స్‌గా దాచుకున్నాం కదా? ఆ జ్యూస్ సైజు 768 (512 క్యామెరా + 256 సెన్సార్).
#         అవుట్‌పుట్ (embedding_dim = 128): 768 సైజు ఉన్న ఆ పొడవాటి డేటాను నేరుగా ట్రాన్స్‌ఫార్మర్ (మెదడు) కి ఇస్తే.. లెక్కలు చేయడానికి చాలా కష్టపడుతుంది (కంప్యూటర్ స్లో అవుతుంది).
#         అందుకే గుమ్మం దగ్గరే ఒక చిన్న లేయర్ (nn.Linear) పెట్టి, ఆ 768 నంబర్లను కుదించి 128 నంబర్లుగా (Embeddings) మార్చేశారు. ఇది అడ్వాన్స్‌డ్ నెట్‌వర్క్‌లలో వాడే చాలా కామన్ ట్రిక్ (దీన్నే Dimensionality Reduction అంటారు).
#
#
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.embedding_dim, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ), 
            num_layers=num_layers
        )
        # Add the fully connected layer
        self.fc2 = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.embedding_dim, 6))
    
#     ════════════════════════════════════════════════════════════════════════════════
#     🕒 Positional Embedding (సమయానుసార గుర్తింపు)
#     ════════════════════════════════════════════════════════════════════════════════
#     వివరణ:
#     ట్రాన్స్‌ఫార్మర్లకి కళ్లు ఉండవు. ఫ్రేమ్స్ అన్నీ ఒకేసారి చూస్తాయి కాబట్టి ఏది ముందు
#     ఏది వెనుక అనే ఆర్డర్ తెలియదు. దానికి ఒక స్టాంప్ లేదా బార్‌కోడ్ వేయడమే ఈ ఫంక్షన్ డ్యూటీ.
#
#     సాధారణ నంబర్లు (0, 1, 2...) ఎందుకు వాడకూడదు?
#     1. వీడియో పెద్దదైతే నంబర్లు (ఉదా: 50,000) కూడా బాగా పెద్దవైపోయి అసలైన డేటాని 
#        డామినేట్ చేస్తాయి. దీనివల్ల న్యూరల్ నెట్‌వర్క్ ఎర్రర్స్ ఇస్తుంది (Exploding values).
#     2. వీడియోల సైజుని బట్టి ఫ్రేమ్‌ల మధ్య సంబంధం/దూరం మారిపోతూ ఉండి మోడల్ కన్‌ఫ్యూజ్ అవుతుంది.
#
#     పరిష్కారం: సైనస్ (Sine), కొసైన్ (Cosine) మ్యాజిక్!
#     ఈ తరంగాలు ఎప్పటికీ -1, 1 మధ్యలనే ఉంటాయి కాబట్టి ఎంత పొడవైన వీడియో ఇచ్చినా
#     ప్రతి ఫ్రేమ్‌కి ఒక యూనిక్ కోడ్ వస్తుంది. దీనివల్ల మోడల్ ఆ ఫ్రేమ్‌లను కచ్చితమైన ఆర్డర్‌లో 
#     చాలా సులభంగా గుర్తుపట్టగలదు!
    def positional_embedding(self, seq_length):
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        # గడియారంలో సెకన్ల ముల్లు ఫాస్ట్‌గా, గంటల ముల్లు నెమ్మదిగా తిరిగినట్లు.. ఈ 128 ఫీచర్లకు 
        # వేర్వేరు వేగాలతో (వేర్వేరు Frequencies తో) తరంగాలను ఇవ్వడమే div_term పని. 
        # దీనివల్ల ఎంత పొడవైన వీడియో ఇచ్చినా, ప్రతి ఫ్రేమ్‌కి ఒక యూనిక్ బార్‌కోడ్ తయారవుతుంది!
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * -(math.log(10000.0) / self.embedding_dim))
        pos_embedding = torch.zeros(seq_length, self.embedding_dim)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        pos_embedding = pos_embedding.unsqueeze(0)
        return pos_embedding

#     ════════════════════════════════════════════════════════════════════════════════
#     🎭 Causal Mask (భవిష్యత్తు చూడకుండా అడ్డుకునే మాస్క్)
#     ════════════════════════════════════════════════════════════════════════════════
#     వివరణ:
#     ట్రాన్స్‌ఫార్మర్ ఎప్పుడైనా గతం చూసి భవిష్యత్తు అంచనా వేయాలి. పొరపాటున భవిష్యత్తు
#     చూసి (కాపీ కొట్టి) ప్రస్తుత ఫ్రేమ్ అవుట్‌పుట్ ఇవ్వకూడదు. కాబట్టి కళ్లకు గంతలు కట్టాలి.
#
#     ఉదాహరణ (4 ఫ్రేమ్స్ ఉన్నప్పుడు తయారుచేసే మాస్క్ ఇలా ఉంటుంది):
#     [  0.0, -inf, -inf, -inf ]   <-- 1వ ఫ్రేమ్ (తనని తను చూసుకోగలదు. మిగతావి చూడలేదు)
#     [  0.0,  0.0, -inf, -inf ]   <-- 2వ ఫ్రేమ్ (1,2 ని చూసుకోగలదు. 3,4 ని చూడలేదు)
#     [  0.0,  0.0,  0.0, -inf ]   <-- 3వ ఫ్రేమ్ (1,2,3 ని చూసుకోగలదు. 4 ని చూడలేదు)
#     [  0.0,  0.0,  0.0,  0.0 ]   <-- 4వ ఫ్రేమ్ (పాతవి కాబట్టి అన్నింటినీ చూసుకోగలదు)
#
#     లాజిక్:
#     * -inf ఉన్న చోట అటెన్షన్ 0% అయిపోతుంది (Softmax మ్యాజిక్ వల్ల). 
#     * 0.0 ఉన్న చోట ఒరిజినల్ పాత స్కోర్ ఏమి మారదు కాబట్టి, ఆ డేటాని వాడుకుంటుంది.
    def generate_square_subsequent_mask(self, sz, device=None, dtype=None):
        """Generate a square causal mask for sequence.
        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        return torch.triu(
                torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
                diagonal=1
        )


    def forward(self, batch, gt):
        visual_inertial_features, _, _ = batch
        seq_length = visual_inertial_features.size(1)

        # 1. పొజిషనల్ ఎంబెడ్డింగ్ ఫంక్షన్‌ నుండి (సైనస్/కొసైన్ బార్‌కోడ్ మ్యాజిక్) బార్‌కోడ్స్ తీసుకోవడం.
        # 2. .to(visual_inertial_features.device): అసలైన కెమెరా డేటా (GPU) ఏ డివైజ్‌లో (గదిలో) ఉందో, 
        #    కొత్తగా తయారైన బార్‌కోడ్‌లను (సాధారణంగా CPU లో తయారవుతాయి) ఖచ్చితంగా అక్కడికే పంపమని అర్థం.
        #    (రెండూ వేర్వేరు గదుల్లో ఉంటే పైటోర్చ్ వాటిని కలపలేదు, కోడ్ క్రాష్ అవుతుంది).
        pos_embedding = self.positional_embedding(seq_length).to(visual_inertial_features.device)
        visual_inertial_features = self.fc1(visual_inertial_features)
        visual_inertial_features += pos_embedding

        
        # Passing through the transformer encoder with the mask
        mask = self.generate_square_subsequent_mask(seq_length, visual_inertial_features.device)
        output = self.transformer_encoder(visual_inertial_features, mask=mask, is_causal=True)
        #output = self.transformer_encoder(visual_inertial_features, mask=None)

        # Pass the output through the fully connected layer
        output = self.fc2(output)

        return output


class TokenizedPoseTransformer(nn.Module):
    def __init__(self,
                 input_dim=768,
                 embedding_dim=128,
                 num_layers=2,
                 nhead=8,
                 dim_feedforward=512,
                 dropout=0.1,
                 scale=1.0,
                 low_limit=-1.0,
                 high_limit=1.0,
                 n_tokens=4096,
                 n_special_tokens=1,
                 pad_token_id=0,
                 eos_token_id=1,
                 use_eos_token=False,
                 context_length=11):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
        )
        self.embeddings = nn.ModuleList([nn.Embedding(n_tokens, embedding_dim) for k in range(6)])
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.embedding_dim, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ), 
            num_layers=num_layers
        )
        # Add the fully connected layer
        self.linears = nn.ModuleList([nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.embedding_dim, n_tokens)) for k in range(6)])

        from chronos_models.chronos import OdometryBins, ChronosConfig

        tokenizer_config = ChronosConfig(n_tokens=n_tokens,
                                         n_special_tokens=n_special_tokens,
                                         context_length=context_length,
                                         pad_token_id=pad_token_id,
                                         eos_token_id=eos_token_id,
                                         use_eos_token=use_eos_token
                                         )

        self.tokenizer = OdometryBins(low_limit=low_limit,
                                      high_limit=high_limit,
                                      config=tokenizer_config)
        self.tokenizer.centers =  self.tokenizer.centers.to("cuda")
        self.tokenizer.boundaries = self.tokenizer.boundaries.to("cuda")
        self.scale = torch.Tensor([scale]).to("cuda")
        self.CELoss = torch.nn.CrossEntropyLoss()
    
    def positional_embedding(self, seq_length):
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * -(math.log(10000.0) / self.embedding_dim))
        pos_embedding = torch.zeros(seq_length, self.embedding_dim)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        pos_embedding = pos_embedding.unsqueeze(0)
        return pos_embedding

    def generate_square_subsequent_mask(self, sz, device=None, dtype=None):
        """Generate a square causal mask for sequence.
        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        return torch.triu(
                torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
                diagonal=1
        )


    def argmax_sampler(self, output_probabilities):
        """
        Sample indices from output probabilities using argmax.
        
        Parameters:
        - output_probabilities (torch.Tensor): A tensor containing output probabilities or logits.
          Shape: [batch_size, sequence_length, num_classes]
        
        Returns:
        - torch.Tensor: A tensor containing the sampled indices (token IDs).
          Shape: [batch_size, sequence_length]
        """
        # Use argmax to get the index of the maximum probability for each token in the sequence
        sampled_indices = torch.argmax(output_probabilities, dim=-1)
    
        return sampled_indices

    def forward(self, batch, gt):
        visual_inertial_features, _, _ = batch
        seq_length = visual_inertial_features.size(1)
        B,S,E = visual_inertial_features.shape
        gt = torch.tensor(gt).view(B,S,6).to("cuda")

        # Generate causal mask
        pos_embedding = self.positional_embedding(seq_length).to(visual_inertial_features.device)
        visual_inertial_features = self.fc1(visual_inertial_features)
        visual_inertial_features += pos_embedding


        B,S,D = gt.shape

        # Generate input tokens
        _tmp = [self.tokenizer.input_transform(gt[:,:,k], scale=self.scale) for k in range(D)]
        # Separate the outputs into individual components
        tokens, attention_masks, scales = zip(*_tmp)
        # Stack each component separately
        tokens = torch.stack(tokens)  # Shape: (6, batch_size, seq_len)
        tokens = tokens.permute(1, 2, 0)  # Shape: (batch_size, seq_len, 6)
        # tokens (batch_size, seq_len, 6)

        input_tokens = torch.roll(tokens, 1, 1) # shift inputs to right to estimate next tokens from previous ones
        input_tokens[:,0,:] = 0 # make first element of every sequence <PAD> token, due to shifting it contained last pose

        # (batch_size, seq_len,D) -> (batch_size, seq_len, embed_size)
        #TODO: Make embeddings list, multiple embeddings via indexingö work here is done, implement in init
        input_embeddings = torch.stack([self.embeddings[k](input_tokens[:,:,k]) for k in range(D)]) 
        input_embeddings = torch.sum(input_embeddings,dim=0)
        input_embeddings += pos_embedding
        

        # concatenate latents with inputs
        tf_input = torch.cat([visual_inertial_features, input_embeddings], dim=1)
        # tf_input (batch_size, 2*seq_len, embed_size)

        # Passing through the transformer encoder with the mask
        mask = self.generate_square_subsequent_mask(int(seq_length*2), visual_inertial_features.device)
        output = self.transformer_encoder(tf_input, mask=mask, is_causal=True)
        #output = self.transformer_encoder(visual_inertial_features, mask=None)

        # Pass the output through the fully connected layer
        logits = torch.stack([self.linears[k](output[:,seq_length:,:]) for k in range(D)])  # [6, B, S, vocab_size]
        logits = logits.permute(1, 2, 0, 3)  # Shape: [B, S, 6, vocab_size]

        B, K, T = tokens.shape
        ce = torch.zeros([], device=tokens.device)
        ce_per_codebook: tp.List[torch.Tensor] = []
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  
            tokens_k = tokens[:, k, ...].contiguous().view(-1)  
            q_ce = self.CELoss(logits_k, tokens_k)
            ce += q_ce
            ce_per_codebook.append(q_ce.detach())
        ce = ce / K

        sampled_indices = self.argmax_sampler(logits) # (batch_size, seq_len,6, num_tokens) -> (batch_size, seq_len,6)

        poses = torch.stack([self.tokenizer.output_transform(sampled_indices[:,:,k], scale=self.scale) for k in range(D)])
        poses = poses.permute(1,2,0) # (6, batch_size, seq_len) -> (batch_size, seq_len, 6)
        assert poses.shape[-1] == 6, 'you need to give 6 dim output'

        return poses, ce.mean()




class PoseTransformerVisual(nn.Module):
    def __init__(self, input_dim=512, embedding_dim=128, num_layers=2, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.embedding_dim, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ), 
            num_layers=num_layers
        )
        # Add the fully connected layer
        self.fc2 = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.embedding_dim, 6))
    
    def positional_embedding(self, seq_length):
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * -(math.log(10000.0) / self.embedding_dim))
        pos_embedding = torch.zeros(seq_length, self.embedding_dim)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        pos_embedding = pos_embedding.unsqueeze(0)
        return pos_embedding

    def generate_square_subsequent_mask(self, sz, device=None, dtype=None):
        """Generate a square causal mask for sequence.
        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        return torch.triu(
                torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
                diagonal=1
        )


    def forward(self, batch, gt):
        visual_inertial_features, _, _ = batch
        visual_inertial_features = visual_inertial_features[:,:,:512]
        seq_length = visual_inertial_features.size(1)

        # Generate causal mask
        pos_embedding = self.positional_embedding(seq_length).to(visual_inertial_features.device)
        visual_inertial_features = self.fc1(visual_inertial_features)
        visual_inertial_features += pos_embedding

        
        # Passing through the transformer encoder with the mask
        mask = self.generate_square_subsequent_mask(seq_length, visual_inertial_features.device)
        output = self.transformer_encoder(visual_inertial_features, mask=mask, is_causal=True)
        #output = self.transformer_encoder(visual_inertial_features, mask=None)

        # Pass the output through the fully connected layer
        output = self.fc2(output)

        return output


class PoseTransformerInertial(nn.Module):
    def __init__(self, input_dim=256, embedding_dim=128, num_layers=2, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.embedding_dim, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ), 
            num_layers=num_layers
        )
        # Add the fully connected layer
        self.fc2 = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.embedding_dim, 6))
    
    def positional_embedding(self, seq_length):
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * -(math.log(10000.0) / self.embedding_dim))
        pos_embedding = torch.zeros(seq_length, self.embedding_dim)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        pos_embedding = pos_embedding.unsqueeze(0)
        return pos_embedding

    def generate_square_subsequent_mask(self, sz, device=None, dtype=None):
        """Generate a square causal mask for sequence.
        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        return torch.triu(
                torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
                diagonal=1
        )


    def forward(self, batch, gt):
        visual_inertial_features, _, _ = batch
        visual_inertial_features = visual_inertial_features[:,:,512:]
        seq_length = visual_inertial_features.size(1)

        # Generate causal mask
        pos_embedding = self.positional_embedding(seq_length).to(visual_inertial_features.device)
        visual_inertial_features = self.fc1(visual_inertial_features)
        visual_inertial_features += pos_embedding

        
        # Passing through the transformer encoder with the mask
        mask = self.generate_square_subsequent_mask(seq_length, visual_inertial_features.device)
        output = self.transformer_encoder(visual_inertial_features, mask=mask, is_causal=True)
        #output = self.transformer_encoder(visual_inertial_features, mask=None)

        # Pass the output through the fully connected layer
        output = self.fc2(output)

        return output
