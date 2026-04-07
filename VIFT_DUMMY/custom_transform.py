# from __future__ import division
import torch
import random
import numpy as np
import torchvision.transforms.functional as TF

# ================================================================================
# 💡 Compose క్లాస్ యొక్క ముఖ్య ఉద్దేశ్యం:
# ================================================================================
# "మనం చేయాల్సిన రకరకాల పనులను (ఉదాహరణకు: ఇమేజ్ సైజ్ తగ్గించడం, కలర్ మార్చడం,
# టెన్సార్గా మార్చడం) విడివిడిగా చేయకుండా, వాటన్నింటినీ ఒక 'లిస్ట్' లాగా కలిపి
# ఒకేసారి వరుసక్రమంలో (Sequential గా) చేయడం."
# ================================================================================

# ── 1. క్లాస్ మరియు ఇనిషియలైజేషన్ (Class & Initialization) ──
# పైథాన్లో Compose అనే ఒక కొత్త క్లాస్ను సృష్టిస్తున్నారు.
# (object) అని రాయడం పాత పైథాన్ 2 పద్ధతి, పైథాన్ 3 లో రాయకపోయినా పర్వాలేదు.
class Compose(object):
    # __init__ అనేది కన్స్ట్రక్టర్ (Constructor).
    # మీరు ఈ Compose ని వాడటం మొదలుపెట్టిన వెంటనే, మొట్టమొదట రన్ అయ్యే ఫంక్షన్ ఇదే.
    # ఇక్కడ transforms అంటే మీరు చేయాలనుకుంటున్న పనుల జాబితా (List).
    # ఉదాహరణకు: [Resize(), ToTensor()]
    def __init__(self, transforms):
        # మీరు ఇచ్చిన ఆ పనుల జాబితాను (లిస్ట్ను), భవిష్యత్తులో వాడుకోవడం కోసం
        # self.transforms అనే వేరియబుల్ లో భద్రంగా దాచిపెట్టుకుంటున్నారు.
        self.transforms = transforms

    # ── 2. ది మ్యాజిక్ ఫంక్షన్ (The Magic Callable Method) ──
    # __call__ అనేది ఒక స్పెషల్/మ్యాజిక్ ఫంక్షన్.
    # సాధారణంగా ఒక క్లాస్ లోని ఫంక్షన్ని పిలవాలంటే object.function_name() అని పిలవాలి.
    # కానీ __call__ వాడితే, ఆ ఆబ్జెక్ట్ను డైరెక్ట్గా ఒక ఫంక్షన్ లాగా వాడేయొచ్చు.
    # (ఉదాహరణకు: నేరుగా transform_train(img, imu, int) అని బ్రాకెట్స్ పెట్టి వాడేయొచ్చు)
    # ఈ ఫంక్షన్ మూడు ఇన్పుట్స్ తీసుకుంటుంది:
    #   - కెమెరా ఇమేజెస్
    #   - సెన్సార్ డేటా (IMU)
    #   - కెమెరా ఇంట్రిన్సిక్స్ (కెమెరా లెన్స్ సెట్టింగ్స్)
    def __call__(self, images, imus, intrinsics):
        # ఇందాక మనం దాచిపెట్టుకున్న పనుల లిస్ట్ (self.transforms) లో ఉన్న
        # ఒక్కో పనిని (t) బయటకు తీస్తూ లూప్ (Loop) తిప్పుతున్నాం.
        for t in self.transforms:
            # 🔑 ఇది అత్యంత ముఖ్యమైన లైన్. ఇక్కడ పైప్లైన్ (Pipeline) జరుగుతోంది.
            # - లూప్లో మొదట వచ్చిన పనికి (ఉదా: Resize కి) పచ్చి డేటాని ఇచ్చారు.
            # - ఆ Resize పని పూర్తయ్యాక వచ్చిన అవుట్పుట్తో పాత values ని Overwrite చేశారు.
            # - ఇప్పుడు లూప్ రెండో పని (ఉదా: Normalize) దగ్గరికి వెళ్ళినప్పుడు..
            #   దానికి పచ్చి డేటా వెళ్ళదు, ఇందాక Resize చేసిన డేటా వెళ్తుంది.
            # - ఇలా ఒకదాని అవుట్పుట్ ఇంకొకదానికి ఇన్పుట్గా వెళ్తూ పనులన్నీ పూర్తవుతాయి.
            images, imus, intrinsics = t(images, imus, intrinsics)
        # లిస్ట్ లో ఉన్న పనులన్నీ (Transforms) అప్లై అయిపోయాక,
        # పూర్తిగా ప్రాసెస్ అయిన ఆ ఫైనల్ డేటాని బయటకు (Return) ఇచ్చేస్తుంది.
        return images, imus, intrinsics

    # ── 3. ప్రింట్ చేసినప్పుడు అందంగా కనిపించడానికి (String Representation) ──
    # __repr__ (Representation) అనేది కూడా ఒక మ్యాజిక్ ఫంక్షన్.
    # మీరు ఎప్పుడైనా కన్సోల్ లో print(transform_train) అని టైప్ చేస్తే,
    # కంప్యూటర్ కు ఏమని ప్రింట్ చేయాలో ఈ ఫంక్షన్ చెబుతుంది.
    def __repr__(self):
        # self.__class__.__name__ అంటే ఆ క్లాస్ పేరు ("Compose") ని డైనమిక్ గా
        # తీసుకుంటుంది. దానికి ఒక బ్రాకెట్ ( ని కలుపుతోంది.
        # (ఇప్పుడు టెక్స్ట్ ఇలా ఉంటుంది: Compose( )
        format_string = self.__class__.__name__ + '('
        # మళ్ళీ మన పనుల లిస్ట్ (Transforms list) లోని ఒక్కో పనిని బయటకు తీస్తోంది.
        for t in self.transforms:
            # ప్రింట్ చేసేటప్పుడు అంతా ఒకే లైన్ లో రాకుండా ఉండటానికి
            # ఒక కొత్త లైన్ (Enter కీ నొక్కినట్టు - Newline) ని యాడ్ చేస్తోంది.
            format_string += '\n'
            # ముందు నాలుగు స్పేస్లు (ఖాళీలు) వదిలి, ఆ తర్వాత ఆ పని (Transform) పేరును
            # ప్రింట్ చేస్తోంది. ఇది చూడటానికి నీట్ గా అలైన్ (Align) అయి కనిపిస్తుంది.
            format_string += '    {0}'.format(t)
        # లిస్ట్ అంతా అయిపోయాక, క్లోజ్ బ్రాకెట్ ) ని యాడ్ చేస్తోంది.
        # ప్రింట్ అయినప్పుడు ఇలా కనిపిస్తుంది:
        #   Compose(
        #       ToTensor()
        #       Resize((256, 512))
        #   )
        format_string += '\n)'
        return format_string

# ================================================================================
# 💡 Normalize క్లాస్ యొక్క ఉద్దేశ్యం:
# ================================================================================
# న్యూరల్ నెట్‌వర్క్ కి ఇమేజ్ ఇచ్చేముందు, పిక్సెల్ విలువలను "సమానమైన స్కేల్" లోకి
# తీసుకురావడం. ఇది లేకపోతే నెట్‌వర్క్ చాలా నెమ్మదిగా నేర్చుకుంటుంది.
#
# 📐 గణిత సూత్రం (Math Formula):
#   normalized_value = (pixel_value - mean) / std
#
# ఉదాహరణ: ఒక పిక్సెల్ Red channel విలువ 200 ఉంటే,
#   mean=0.485, std=0.229 అయితే:
#   (200/255 - 0.485) / 0.229 = (0.784 - 0.485) / 0.229 = 1.306
#   ఇప్పుడు విలువ -3 నుంచి +3 రేంజ్ లో ఉంటుంది (బాగా నేర్చుకోవడానికి అనువైనది!)
# ================================================================================

# ── Normalize: ఇమేజ్ పిక్సెల్ విలువలను స్టాండర్డైజ్ చేసే క్లాస్ ──
class Normalize(object):
    # కన్స్ట్రక్టర్: mean మరియు std విలువలను తీసుకుంటుంది.
    # mean = సగటు విలువ (ఉదా: ImageNet కోసం [0.485, 0.456, 0.406] — R, G, B channels)
    # std  = విస్తరణ (ఉదా: ImageNet కోసం [0.229, 0.224, 0.225] — R, G, B channels)
    def __init__(self, mean, std):
        self.mean = mean  # భవిష్యత్తులో వాడుకోవడానికి mean ని దాచిపెట్టారు.
        self.std = std    # భవిష్యత్తులో వాడుకోవడానికి std ని దాచిపెట్టారు.

    # మ్యాజిక్ కాల్ మెథడ్: ఇమేజెస్, IMU, intrinsics తీసుకుంటుంది.
    # (IMU మరియు intrinsics ని ఏమీ మార్చదు, కేవలం ఇమేజెస్ ని మాత్రమే normalize చేస్తుంది)
    def __call__(self, images, imus, intrinsics):
        # images లో ఉన్న ప్రతి ఇమేజ్ టెన్సర్ ని ఒక్కొక్కటిగా తీస్తున్నాం.
        # (ఉదా: 11 ఫ్రేమ్ల సీక్వెన్స్ ఉంటే, 11 సార్లు లూప్ తిరుగుతుంది)
        for tensor in images:
            # 🔑 ఇది అత్యంత ముఖ్యమైన లైన్!
            # ఒక ఇమేజ్ టెన్సర్ లో 3 చానెల్స్ ఉంటాయి: Red (R), Green (G), Blue (B).
            # zip(tensor, self.mean, self.std) అంటే:
            #   t = tensor[0] (Red channel),   m = mean[0] (0.485), s = std[0] (0.229)
            #   t = tensor[1] (Green channel), m = mean[1] (0.456), s = std[1] (0.224)
            #   t = tensor[2] (Blue channel),  m = mean[2] (0.406), s = std[2] (0.225)
            # ఇలా ఒక్కో channel కి దాని సొంత mean, std వాడి normalize చేస్తోంది.
            for t, m, s in zip(tensor, self.mean, self.std):
                # t.sub_(m) = t నుండి mean తీసేయి (in-place, అంటే కొత్త కాపీ చేయకుండా
                #             ఒరిజినల్ డేటానే మార్చేస్తుంది — మెమరీ ఆదా!)
                # .div_(s) = ఆ రిజల్ట్ ని std తో భాగించు (ఇది కూడా in-place)
                # "_" (underscore) ఉన్న ఫంక్షన్లు PyTorch లో "in-place operations"
                # అంటే: t = (t - m) / s  ← ఇదే జరుగుతోంది, కానీ ఎక్స్ట్రా మెమరీ లేకుండా!
                t.sub_(m).div_(s)
        # Normalize అయిన ఇమేజెస్ ని, మార్పు లేని IMU & intrinsics తో కలిపి రిటర్న్ చేస్తోంది.
        return images, imus, intrinsics

    # ప్రింట్ చేసినప్పుడు అందంగా కనిపించడానికి:
    # ఉదా: Normalize(mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'mean: {}, '.format(self.mean)
        format_string += 'std: {})\n'.format(self.std)
        return format_string

# ================================================================================
# 💡 ToTensor క్లాస్ యొక్క ఉద్దేశ్యం:
# ================================================================================
# PIL ఇమేజ్ లేదా NumPy Array ని PyTorch Tensor గా మార్చడం.
# న్యూరల్ నెట్‌వర్క్ కేవలం Tensor లనే అర్థం చేసుకోగలదు, ఇమేజ్ ఫైల్స్ ని కాదు!
#
# 📐 అదనపు గణితం: TF.to_tensor() చేస్తే విలువలు 0~1 రేంజ్ లో వస్తాయి.
#   దాని నుండి 0.5 తీసేయడం వల్ల విలువలు -0.5 ~ +0.5 రేంజ్ లోకి వస్తాయి.
#   ఇది "zero-centering" అంటారు — నెట్‌వర్క్ కి ఇది చాలా మంచిది!
# ================================================================================

# ── ToTensor: ఇమేజ్ ని అంకెల టేబుల్ (Tensor) గా మార్చే క్లాస్ ──
class ToTensor(object):
    # ఈ క్లాస్ కి __init__ అవసరం లేదు ఎందుకంటే ఇది ఏ సెట్టింగ్స్ లేకుండా పని చేస్తుంది.
    def __call__(self, images, imus, gts):
        # ఖాళీ లిస్ట్ ని సృష్టించారు — ఇందులో కన్వర్ట్ అయిన టెన్సర్లు దాచుకుంటాం.
        tensors = []
        # ప్రతి ఇమేజ్ ని ఒక్కొక్కటిగా తీసుకుంటున్నాం.
        for im in images:
            # np.array(im) = PIL Image ని NumPy Array గా మార్చు.
            # TF.to_tensor() = NumPy Array ని PyTorch Tensor గా మార్చు (0~1 రేంజ్).
            # - 0.5 = విలువలను -0.5 ~ +0.5 కి సెంటర్ చేయి (zero-centering).
            # చివరగా ఆ tensor ని లిస్ట్ లో యాడ్ చేస్తోంది.
            tensors.append(TF.to_tensor(np.array(im))- 0.5)
        # torch.stack() = విడివిడిగా ఉన్న టెన్సర్లను ఒక పెద్ద బ్యాచ్ టెన్సర్ గా కలుపుతోంది.
        # ఉదా: 11 ఫ్రేమ్లు ఉంటే [11, 3, H, W] షేప్ లో వస్తుంది.
        tensors = torch.stack(tensors, 0)
        return tensors, imus, gts

    def __repr__(self):
        return self.__class__.__name__ + '()'

# ================================================================================
# 💡 Resize క్లాస్ యొక్క ఉద్దేశ్యం:
# ================================================================================
# వేర్వేరు సైజుల ఇమేజ్‌లను ఒకే ఫిక్స్డ్ సైజ్ కి మార్చడం.
# న్యూరల్ నెట్‌వర్క్ కి ఇన్పుట్ సైజ్ ఎప్పుడూ ఒకేలా ఉండాలి (ఉదా: 256×512).
# లేకపోతే matrix multiplication crash అవుతుంది!
# ================================================================================

# ── Resize: అన్ని ఇమేజెస్ ని ఒకే సైజ్ కి కుదించే/పెంచే క్లాస్ ──
class Resize(object):
    # size = (height, width) ఫార్మాట్ లో ఉంటుంది.
    # default: (256, 512) — ఎత్తు 256 పిక్సెల్స్, వెడల్పు 512 పిక్సెల్స్.
    def __init__(self, size=(256, 512)):
        self.size = size

    def __call__(self, images, imus, gts):
        # ప్రతి ఇమేజ్ కి TF.resize() వాడి కొత్త సైజ్ కి మారుస్తోంది.
        # antialias=True → సైజ్ తగ్గించేటప్పుడు ఇమేజ్ "బ్లాక్స్" లాగా (pixelated)
        #   కనిపించకుండా, స్మూత్ గా కనిపించేలా చేస్తుంది.
        tensors = [TF.resize(im, size=self.size, antialias=True) for im in images]
        # మళ్ళీ అన్నింటినీ ఒకే బ్యాచ్ టెన్సర్ గా కలుపుతోంది.
        tensors = torch.stack(tensors, 0)
        return tensors, imus, gts

    # ప్రింట్: Resize(img_h: 256, img_w: 512)
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'img_h: {}, '.format(self.size[0])
        format_string += 'img_w: {})'.format(self.size[1])
        return format_string

# ================================================================================
# 💡 RandomHorizontalFlip క్లాస్ యొక్క ఉద్దేశ్యం:
# ================================================================================
# 50% ఛాన్స్ తో ఇమేజ్ ని అద్దంలో చూసినట్లు తిప్పేయడం (Mirror Flip) 🪞
#
# ⚠️ ముఖ్యమైన పాయింట్: ఇమేజ్ ని తిప్పితే, IMU సెన్సర్ డేటా మరియు Ground Truth
# పోజ్ కూడా తిరగేయాల్సి వస్తుంది! లేకపోతే నెట్‌వర్క్ "ఎడమ" ని "కుడి" అని తప్పుడు
# దిశలు నేర్చుకుంటుంది.
#
# 🚗 రియల్ వరల్డ్ అనాలజీ: మీరు కారు ఎడమ వైపు తిరిగారు. ఇప్పుడు ఇమేజ్ ని
# ఫ్లిప్ చేస్తే "కుడి వైపు తిరిగినట్లు" కనిపిస్తుంది. అందుకే IMU లో direction
# signs (-) కి మారాల్సి వస్తుంది!
# ================================================================================

# ── RandomHorizontalFlip: రాండమ్ గా ఇమేజ్ ని అడ్డంగా తిప్పే క్లాస్ ──
class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""
    # p = ఫ్లిప్ అయ్యే probability (ఛాన్స్). default 0.5 అంటే 50% సమయాల్లో ఫ్లిప్ అవుతుంది.
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images, imus, gts):
        # random.random() → 0.0 నుంచి 1.0 మధ్య రాండమ్ నంబర్ ఇస్తుంది.
        # అది self.p (0.5) కంటే తక్కువ ఉంటే → ఫ్లిప్ చేస్తాం!
        if random.random() < self.p:
            # TF.hflip() = Horizontal Flip. ప్రతి ఇమేజ్ ని అడ్డంగా తిప్పేస్తోంది.
            tensors = [TF.hflip(im) for im in images]
            tensors = torch.stack(tensors, 0)

            # 🔑 ఇది చాలా ముఖ్యం — IMU డేటాని కూడా ఫ్లిప్ చేయడం!
            # IMU లో 6 విలువలు ఉంటాయి: [ax, ay, az, wx, wy, wz]
            #   ax = X-axis acceleration, ay = Y-axis, az = Z-axis
            #   wx = X-axis angular velocity, wy = Y-axis, wz = Z-axis
            #
            # ఇమేజ్ ఫ్లిప్ అయినప్పుడు Y-axis (అడ్డం) దిశ మారుతుంది, కాబట్టి:
            #   imus[:, 1] (ay) → sign ఫ్లిప్ (-ay)
            #   imus[:, 3] (wx) → sign ఫ్లిప్ (-wx)
            #   imus[:, 5] (wz) → sign ఫ్లిప్ (-wz)
            imus[:, 1], imus[:, 3], imus[:, 5] = -imus[:, 1], -imus[:, 3], -imus[:, 5]

            # Ground Truth poses కూడా ఫ్లిప్ చేయాలి!
            # gts[:, 1] (Y-translation) → sign ఫ్లిప్
            # gts[:, 2], gts[:, 3] (Rotation angles) → sign ఫ్లిప్
            gts[:, 1], gts[:, 2], gts[:, 3] = -gts[:, 1], -gts[:, 2], -gts[:, 3]
        else:
            # ఫ్లిప్ అవ్వకపోతే, ఇమేజ్‌లను అలాగే (as-is) ఉంచేస్తాం.
            tensors = images
        return tensors, imus, gts

    # ప్రింట్: RandomHorizontalFlip(p: 0.5)
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'p: {})'.format(self.p)
        return format_string

# ================================================================================
# 💡 RandomColorAug క్లాస్ యొక్క ఉద్దేశ్యం:
# ================================================================================
# ట్రైనింగ్ లో ఇమేజ్‌ల రంగులు, ప్రకాశం (brightness), గామా ని రాండమ్ గా మార్చడం. 🎨
#
# 🌤️ ఎందుకంటే: రియల్ వరల్డ్ లో కారు పగటి వెలుగులో, రాత్రి చీకటిలో, మేఘాల నీడలో,
# సూర్యాస్తమయంలో — అన్ని రకాల లైటింగ్ కండిషన్లలో నడుస్తుంది. మోడల్ ఈ వేర్వేరు
# లైటింగ్ కండిషన్లన్నింటిలోనూ బాగా పనిచేయాలంటే, ట్రైనింగ్ లో ఇమేజ్‌ల రంగులను
# కృత్రిమంగా మార్చాలి. దీనినే "Color Augmentation" అంటారు!
#
# 📐 మూడు రకాల మార్పులు జరుగుతాయి:
#   1. Gamma: ఇమేజ్ contrast ని మార్చడం (img^gamma)
#   2. Brightness: మొత్తం ఇమేజ్ ని ప్రకాశవంతంగా/చీకటిగా చేయడం
#   3. Color shift: R, G, B చానెల్ విలువలను విడివిడిగా మార్చడం
# ================================================================================

# ── RandomColorAug: రాండమ్ కలర్/బ్రైట్‌నెస్ మార్పులు చేసే క్లాస్ ──
class RandomColorAug(object):
    # augment_parameters = [gamma_low, gamma_high, brightness_low, brightness_high, color_low, color_high]
    # ఈ 6 నంబర్లు ఏ రేంజ్ లో మార్పులు చేయాలో నిర్ణయిస్తాయి.
    def __init__(self, augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2], p=0.5):
        self.gamma_low = augment_parameters[0]       # 0.8 — Gamma minimum (తక్కువ contrast)
        self.gamma_high = augment_parameters[1]      # 1.2 — Gamma maximum (ఎక్కువ contrast)
        self.brightness_low = augment_parameters[2]  # 0.5 — చీకటి (dark)
        self.brightness_high = augment_parameters[3] # 2.0 — ప్రకాశవంతం (bright)
        self.color_low = augment_parameters[4]       # 0.8 — కలర్ తగ్గించడం
        self.color_high = augment_parameters[5]      # 1.2 — కలర్ పెంచడం
        self.p = p  # 50% ఛాన్స్ తో ఈ augmentation అప్లై అవుతుంది.

    def __call__(self, images, imus, gts):
        # రాండమ్ నంబర్ 0.5 కంటే తక్కువ ఉంటేనే color augmentation చేస్తాం.
        if random.random() < self.p:
            # ఇందాక ToTensor లో -0.5 చేశాం కదా, ఇప్పుడు +0.5 చేసి
            # విలువలను 0~1 రేంజ్ కి తీసుకొస్తున్నాం (gamma operation కోసం).
            images = images + 0.5

            # 0.8 ~ 1.2 మధ్య ఒక రాండమ్ gamma విలువ ఎంపిక.
            random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
            # 0.5 ~ 2.0 మధ్య ఒక రాండమ్ brightness విలువ ఎంపిక.
            random_brightness = np.random.uniform(self.brightness_low, self.brightness_high)
            # R, G, B చానెల్స్ కి విడివిడిగా 0.8 ~ 1.2 మధ్య 3 రాండమ్ కలర్ విలువలు.
            random_colors = np.random.uniform(self.color_low, self.color_high, 3)
            
            # ── Step 1: Gamma Correction ──
            # img^gamma → gamma < 1 అయితే ఇమేజ్ లైట్ అవుతుంది,
            #             gamma > 1 అయితే ఇమేజ్ డార్క్ అవుతుంది.
            # ఇది ఇమేజ్ "contrast" ని మారుస్తుంది.
            img_aug = images ** random_gamma
            
            # ── Step 2: Brightness Shift ──
            # మొత్తం ఇమేజ్ ని ఒక రాండమ్ నంబర్ తో multiply చేయడం.
            # 0.5 అయితే సగం చీకటి, 2.0 అయితే రెట్టింపు ప్రకాశం!
            img_aug = img_aug * random_brightness
            
            # ── Step 3: Color Channel Shift ──
            # ప్రతి కలర్ చానెల్ (R, G, B) ని విడివిడిగా ఒక రాండమ్ నంబర్ తో multiply.
            # ఇది ఇమేజ్ యొక్క "టింట్ (tint)" ని మారుస్తుంది.
            # ఉదా: Red channel ని 1.2 తో multiply → ఇమేజ్ కొంచెం ఎర్రగా కనిపిస్తుంది.
            for i in range(3):
                # [:, i, :, :] → i=0 Red, i=1 Green, i=2 Blue channel.
                img_aug[:, i, :, :] *= random_colors[i]
            
            # ── Step 4: Saturation (Clamping) ──
            # పైన చేసిన మార్పుల వల్ల విలువలు 0~1 రేంజ్ దాటిపోతే ఇబ్బంది.
            # torch.clamp(img, 0, 1) → 0 కంటే తక్కువ ఉంటే 0 కి, 1 కంటే ఎక్కువ ఉంటే 1 కి కట్ చేస్తుంది.
            # చివరగా మళ్ళీ -0.5 చేసి -0.5 ~ +0.5 రేంజ్ కి తీసుకొస్తున్నాం (zero-centered).
            img_aug = torch.clamp(img_aug, 0, 1) - 0.5

        else:
            # ఈసారి augmentation అవ్వలేదు → ఇమేజ్‌లను అలాగే (as-is) ఉంచేస్తాం.
            img_aug = images

        return img_aug, imus, gts

    # ప్రింట్: RandomColorAug(gamma: 0.8-1.2, brightness: 0.5-2.0, color shift: 0.8-1.2, p: 0.5)
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'gamma: {}-{}, '.format(self.gamma_low, self.gamma_high)
        format_string += 'brightness: {}-{}, '.format(self.brightness_low, self.brightness_high)
        format_string += 'color shift: {}-{}, '.format(self.color_low, self.color_high)
        format_string += 'p: {})'.format(self.p)
        return format_string
