import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import scipy.io as sio
from path import Path
from kitti_utils import rotationError, read_pose_from_text
import custom_transform
from collections import Counter
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d

IMU_FREQ = 10
class KITTI(Dataset):
    def __init__(self, root,
                 sequence_length=11,
                 train_seqs=['00', '01', '02', '04', '06', '08', '09'],
                 transform=None):
        
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.transform = transform
        self.train_seqs = train_seqs
        self.make_dataset()
    
    def make_dataset(self):
        sequence_set = []
        for folder in self.train_seqs:
            poses, poses_rel = read_pose_from_text(self.root/'poses/{}.txt'.format(folder))
            imus = sio.loadmat(self.root/'imus/{}.mat'.format(folder))['imu_data_interp']
            # IMPORTANT: image_2 is the KITTI convention for the left color camera images.
            # (KITTI uses image_0/image_1 for grayscale stereo and image_2/image_3 for color stereo.)
            fpaths = sorted((self.root/'sequences/{}/image_2'.format(folder)).files("*.png"))      
            for i in range(len(fpaths)-self.sequence_length):
                img_samples = fpaths[i:i+self.sequence_length]
                imu_samples = imus[i*IMU_FREQ:(i+self.sequence_length-1)*IMU_FREQ+1]
                pose_samples = poses[i:i+self.sequence_length]
                pose_rel_samples = poses_rel[i:i+self.sequence_length-1]
                segment_rot = rotationError(pose_samples[0], pose_samples[-1])
                sample = {'imgs':img_samples, 'imus':imu_samples, 'gts': pose_rel_samples, 'rot': segment_rot}
                sequence_set.append(sample)
        self.samples = sequence_set
        
        # Generate weights based on the rotation of the training segments
        # Weights are calculated based on the histogram of rotations according to the method in https://github.com/YyzHarry/imbalanced-regression
        #
        # ముఖ్య ఉద్దేశ్యం: ట్రైనింగ్ డేటాలో ఏ రొటేషన్ యాంగిల్స్ (Rotation angles) అయితే చాలా తక్కువగా ఉన్నాయో,
        # వాటికి ఎక్కువ వెయిటేజ్ (ప్రాముఖ్యత) ఇవ్వడం; ఎక్కువగా ఉన్నవాటికి తక్కువ వెయిటేజ్ ఇవ్వడం.
        #
        # 📊 విజువల్ డయాగ్రమ్ (The Process Flow):
        #
        # [Step 1: Raw Data] -> రొటేషన్ యాంగిల్స్ (ఉదా: 10°, 15°, 45°, 80°, 85°)
        #       │
        #       ▼
        # [Step 2: Bins]     -> యాంగిల్స్ను 10 భాగాలుగా (Bins) విడగొట్టడం.
        #                       [0°-10°] | [11°-20°] | ... | [81°-90°]
        #       │
        #       ▼
        # [Step 3: Empirical] -> ఏ బిన్లో ఎన్ని శాంపిల్స్ ఉన్నాయో లెక్కించడం (Histogram).
        # [Distribution]        Bin 1: ▮▮▮▮▮ (50)  <-- మెజారిటీ డేటా
        #                       Bin 2: ▮ (10)
        #                       Bin 3: (0)         <-- డేటా లేదు!
        #                       Bin 4: ▮▮ (20)
        #       │
        #       ▼
        # [Step 4: Gaussian] -> Gaussian Kernel ఉపయోగించి ఖాళీగా ఉన్న Bin 3 కి,
        # [Smoothing (LDS)]     పక్కన ఉన్న Bins (2 & 4) నుండి కొంత వాల్యూను పంచుతుంది. (ఇదే Smoothing).
        #                       Bin 1: ▮▮▮▮ (45)
        #                       Bin 2: ▮▮ (15)
        #                       Bin 3: ▮ (5)       <-- ఇప్పుడు 0 కాదు! (Smoothed)
        #                       Bin 4: ▮▮ (18)
        #       │
        #       ▼
        # [Step 5: Weights]  -> ఫార్ములా: Weight = 1 / (Smoothed Value)
        #                       ఎక్కువ డేటా ఉన్న Bin 1 కి -> తక్కువ వెయిట్ (1/45 = 0.02)
        #                       తక్కువ డేటా ఉన్న Bin 3 కి -> ఎక్కువ వెయిట్ (1/5 = 0.20)
        #                       (తద్వారా రేర్ డేటా మీద AI మోడల్ ఎక్కువ ఫోకస్ చేస్తుంది).
        #
        # Step 1: డేటాను సేకరించడం మరియు మార్చడం
        # rot రేడియన్స్ లో ఉంది → *180/np.pi ద్వారా డిగ్రీలలోకి మారుస్తున్నారు.
        # ఆ తర్వాత క్యూబ్ రూట్ (np.cbrt) అప్లై చేసి rot_list తయారు చేస్తున్నారు.
        rot_list = np.array([np.cbrt(item['rot']*180/np.pi) for item in self.samples])
        # Step 2: మినిమమ్ నుండి మాక్సిమమ్ వరకు 10 సమాన భాగాలుగా (Bins) విభజిస్తున్నారు.
        rot_range = np.linspace(np.min(rot_list), np.max(rot_list), num=10)
        # np.digitize ద్వారా ప్రతి శాంపిల్ ఏ బిన్ లోకి వెళ్తుందో ఆ ఇండెక్స్ నంబర్ కనుక్కుంటున్నారు.
        indexes = np.digitize(rot_list, rot_range, right=False)
        # Step 3: ప్రతి బిన్లో ఎన్ని శాంపిల్స్ ఉన్నాయో లెక్కించడం (Empirical Label Distribution).
        # ఏ బిన్లోనైనా డేటా లేకపోతే, అక్కడ 0 పెడుతుంది.
        num_samples_of_bins = dict(Counter(indexes))
        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(1, len(rot_range)+1)]

        # Step 4: LDS (Label Distribution Smoothing) అప్లై చేయడం.
        # Gaussian Kernel (సైజు 7, సిగ్మా 5) క్రియేట్ చేసి, హిస్టోగ్రామ్ మీద convolve1d చేస్తున్నారు.
        # దీనివల్ల ఖాళీ బిన్స్కి పక్కన ఉన్న బిన్స్ నుండి కొంత వాల్యూ పంచబడి గ్రాఫ్ స్మూత్గా మారుతుంది.
        # దీన్నే "Effective Label Distribution" అంటారు.
        lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=7, sigma=5)
        eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')

        # Step 5: వెయిట్స్ క్యాలిక్యులేట్ చేయడం. ఫార్ములా: Weight = 1 / (ఆ బిన్ యొక్క Effective Density).
        # ఎక్కువ ఫోటోలు ఉన్న బిన్కి తక్కువ వెయిట్, రేర్ యాంగిల్స్కి ఎక్కువ వెయిట్.
        # తద్వారా AI మోడల్ ఇంబ్యాలెన్స్డ్ డేటాను పర్ఫెక్ట్గా నేర్చుకుంటుంది.
        self.weights = [np.float32(1/eff_label_dist[bin_idx-1]) for bin_idx in indexes]

    def __getitem__(self, index):
        sample = self.samples[index]
        imgs = [np.asarray(Image.open(img)) for img in sample['imgs']]
        
        if self.transform is not None:
            imgs, imus, gts = self.transform(imgs, np.copy(sample['imus']), np.copy(sample['gts']))
        else:
            imus = np.copy(sample['imus'])
            gts = np.copy(sample['gts']).astype(np.float32)
        
        rot = sample['rot'].astype(np.float32)
        weight = self.weights[index]

        return (imgs.to(torch.float), torch.from_numpy(imus).to(torch.float), rot, weight), torch.from_numpy(gts).to(torch.float)

    def __len__(self):
        return len(self.samples)

    # __repr__ అనేది Python special method. print(dataset) చేసినప్పుడు ఏమి చూపించాలో ఇది నిర్ణయిస్తుంది.
    # ఇది డీబగ్గింగ్ మరియు లాగింగ్ కోసం మాత్రమే — ట్రైనింగ్ మీద ఏ ప్రభావం ఉండదు.
    #
    # ఉదాహరణ అవుట్పుట్:
    # Dataset KITTI
    #     Training sequences: 00 01 02 04 06 08 09
    #     Number of segments: 15720
    #     Transforms (if any): Compose(
    #                               Resize((256, 512))
    #                               Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #                           )
    def __repr__(self):
        # __class__.__name__ ద్వారా క్లాస్ పేరు డైనమిక్గా తీసుకుంటుంది (ఉదా: "Dataset KITTI")
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        # ట్రైనింగ్ సీక్వెన్స్ల పేర్లను జాబితాగా చూపిస్తుంది (ఉదా: "00 01 02 04 06 08 09")
        fmt_str += '    Training sequences: '
        for seq in self.train_seqs:
            fmt_str += '{} '.format(seq)
        fmt_str += '\n'
        # మొత్తం ట్రైనింగ్ సెగ్మెంట్ల సంఖ్య చూపిస్తుంది
        fmt_str += '    Number of segments: {}\n'.format(self.__len__())
        # ట్రాన్స్ఫార్మ్ యొక్క __repr__ ని ప్రింట్ చేస్తుంది.
        # .replace('\n', '\n' + ' ' * len(tmp)) ద్వారా మల్టీ-లైన్ అవుట్పుట్ సరిగ్గా ఇండెంట్ అవుతుంది,
        # తద్వారా కంటిన్యుయేషన్ లైన్లు మొదటి లైన్ కింద అలైన్ అవుతాయి.
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str


# get_lds_kernel_window: Label Distribution Smoothing (LDS) లో అత్యంత కీలకమైన భాగం.
# ఒక డేటా పాయింట్ దగ్గర ఉన్న సమాచారాన్ని (కౌంట్ను), దాని చుట్టుపక్కల ఉన్న వాల్యూస్కి
# "ఎలా పంచిపెట్టాలి (Smooth చేయాలి)" అని నిర్ణయించే ఒక "ఫిల్టర్/విండో" ను ఈ ఫంక్షన్ తయారు చేస్తుంది.
#
# 📥 ఇన్పుట్ పారామీటర్స్:
#   kernel: ఏ పద్ధతిలో స్మూత్ చేయాలి? ('gaussian', 'triang', లేదా 'laplace' మాత్రమే)
#   ks (Kernel Size): విండో సైజు (ఉదా: 5 లేదా 7). ఎక్కువ ఉంటే డేటా అంత దూరం పంచిపెట్టబడుతుంది.
#                     సాధారణంగా బేసి సంఖ్య (Odd number) అయి ఉంటుంది.
#   sigma: కర్వ్ ఎంత వెడల్పుగా ఉండాలో నిర్ణయించే విలువ (Standard deviation).
#
def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    # సెంటర్ పాయింట్ కనుక్కోవడం: ks=7 అయితే half_ks=3.
    # మధ్యలో 1 పాయింట్, ఎడమ 3, కుడి 3. (మొత్తం 3+1+3 = 7)
    half_ks = (ks - 1) // 2
    # Gaussian (గౌసియన్) కర్వ్:
    # మొదట మధ్యలో 1.0 పెట్టి అటూ ఇటూ 0లు పెడతారు: [0, 0, 0, 1, 0, 0, 0]
    # gaussian_filter1d ద్వారా ఆ 1.0 ను చుట్టుపక్కల 0లలోకి బ్లర్ (స్ప్రెడ్) చేస్తారు.
    # ఇది గంట ఆకారంలో (Bell Curve) పంపిణీ అవుతుంది.
    # చివరన / max(...) తో భాగించి సెంటర్ వాల్యూని 1.0 కి నార్మలైజ్ చేస్తారు.
    # ఆకారం: ▂ ▄ ▆ █ ▆ ▄ ▂ (స్మూత్ గా పడిపోతుంది)
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    # Triang (ట్రయాంగిల్/త్రిభుజం) కర్వ్:
    # త్రిభుజం ఆకారంలో విండోను ఇస్తుంది. స్ప్రెడ్ సమాన నిష్పత్తిలో (Linear గా) తగ్గుతుంది.
    # ఆకారం: ▃ ▅ ▆ █ ▆ ▅ ▃ (స్ట్రెయిట్ గా పడిపోతుంది)
    elif kernel == 'triang':
        kernel_window = triang(ks)
    # Laplace (లాప్లాస్) కర్వ్:
    # గౌసియన్తో పోలిస్తే సెంటర్ దగ్గర చాలా షార్ప్గా (Pointy) ఉంటుంది.
    # గణిత సూత్రం: e^(-|x|/σ) / 2σ
    # np.arange(-3, 4) (-3 నుండి 3 వరకు) ఉన్న సంఖ్యలకు ఈ ఫార్ములా అప్లై చేసి,
    # max తో భాగించి సెంటర్ వాల్యూని 1.0 కి నార్మలైజ్ చేస్తారు.
    # ఆకారం: ▂ ▄ █ ▄ ▂ (చాలా వేగంగా పడిపోతుంది)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window



