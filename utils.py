#  Tools
"""
    Str_Weak_err： Compute Strong and Weak error。
    data_show_random：
    Com_Weak_Strong_err：
    create_directory_if_not_exists：
"""
import numpy as np
import pandas as pd
import datetime
import os.path
# from Data_get import Get_van_der_pol_data
import random
import torch
import matplotlib.pyplot as plt


def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def Mycolor():
    """
    Colors
    """
    cnames = {
        'darkblue': '#00008B',
        'aliceblue': '#F0F8FF',
        'antiquewhite': '#FAEBD7',
        'aqua': '#00FFFF',
        'aquamarine': '#7FFFD4',
        'azure': '#F0FFFF',
        'beige': '#F5F5DC',
        'bisque': '#FFE4C4',
        'black': '#000000',
        'blanchedalmond': '#FFEBCD',
        'blue': '#0000FF',
        'blueviolet': '#8A2BE2',
        'brown': '#A52A2A',
        'burlywood': '#DEB887',
        'cadetblue': '#5F9EA0',
        'chartreuse': '#7FFF00',
        'chocolate': '#D2691E',
        'coral': '#FF7F50',
        'cornflowerblue': '#6495ED',
        'cornsilk': '#FFF8DC',
        'crimson': '#DC143C',
        'cyan': '#00FFFF',
        'darkcyan': '#008B8B',
        'darkgoldenrod': '#B8860B',
        'darkgray': '#A9A9A9',
        'darkgreen': '#006400',
        'darkkhaki': '#BDB76B',
        'darkmagenta': '#8B008B',
        'darkolivegreen': '#556B2F',
        'darkorange': '#FF8C00',
        'darkorchid': '#9932CC',
        'darkred': '#8B0000',
        'darksalmon': '#E9967A',
        'darkseagreen': '#8FBC8F',
        'darkslateblue': '#483D8B',
        'darkslategray': '#2F4F4F',
        'darkturquoise': '#00CED1',
        'darkviolet': '#9400D3',
        'deeppink': '#FF1493',
        'deepskyblue': '#00BFFF',
        'dimgray': '#696969',
        'dodgerblue': '#1E90FF',
        'firebrick': '#B22222',
        'floralwhite': '#FFFAF0',
        'forestgreen': '#228B22',
        'fuchsia': '#FF00FF',
        'gainsboro': '#DCDCDC',
        'ghostwhite': '#F8F8FF',
        'gold': '#FFD700',
        'goldenrod': '#DAA520',
        'gray': '#808080',
        'green': '#008000',
        'greenyellow': '#ADFF2F',
        'honeydew': '#F0FFF0',
        'hotpink': '#FF69B4',
        'indianred': '#CD5C5C',
        'indigo': '#4B0082',
        'ivory': '#FFFFF0',
        'khaki': '#F0E68C',
        'lavender': '#E6E6FA',
        'lavenderblush': '#FFF0F5',
        'lawngreen': '#7CFC00',
        'lemonchiffon': '#FFFACD',
        'lightblue': '#ADD8E6',
        'lightcoral': '#F08080',
        'lightcyan': '#E0FFFF',
        'lightgoldenrodyellow': '#FAFAD2',
        'lightgreen': '#90EE90',
        'lightgray': '#D3D3D3',
        'lightpink': '#FFB6C1',
        'lightsalmon': '#FFA07A',
        'lightseagreen': '#20B2AA',
        'lightskyblue': '#87CEFA',
        'lightslategray': '#778899',
        'lightsteelblue': '#B0C4DE',
        'lightyellow': '#FFFFE0',
        'lime': '#00FF00',
        'limegreen': '#32CD32',
        'linen': '#FAF0E6',
        'magenta': '#FF00FF',
        'maroon': '#800000',
        'mediumaquamarine': '#66CDAA',
        'mediumblue': '#0000CD',
        'mediumorchid': '#BA55D3',
        'mediumpurple': '#9370DB',
        'mediumseagreen': '#3CB371',
        'mediumslateblue': '#7B68EE',
        'mediumspringgreen': '#00FA9A',
        'mediumturquoise': '#48D1CC',
        'mediumvioletred': '#C71585',
        'midnightblue': '#191970',
        'mintcream': '#F5FFFA',
        'mistyrose': '#FFE4E1',
        'moccasin': '#FFE4B5',
        'navajowhite': '#FFDEAD',
        'navy': '#000080',
        'oldlace': '#FDF5E6',
        'olive': '#808000',
        'olivedrab': '#6B8E23',
        'orange': '#FFA500',
        'orangered': '#FF4500',
        'orchid': '#DA70D6',
        'palegoldenrod': '#EEE8AA',
        'palegreen': '#98FB98',
        'paleturquoise': '#AFEEEE',
        'palevioletred': '#DB7093',
        'papayawhip': '#FFEFD5',
        'peachpuff': '#FFDAB9',
        'peru': '#CD853F',
        'pink': '#FFC0CB',
        'plum': '#DDA0DD',
        'powderblue': '#B0E0E6',
        'purple': '#800080',
        'red': '#FF0000',
        'rosybrown': '#BC8F8F',
        'royalblue': '#4169E1',
        'saddlebrown': '#8B4513',
        'salmon': '#FA8072',
        'sandybrown': '#FAA460',
        'seagreen': '#2E8B57',
        'seashell': '#FFF5EE',
        'sienna': '#A0522D',
        'silver': '#C0C0C0',
        'skyblue': '#87CEEB',
        'slateblue': '#6A5ACD',
        'slategray': '#708090',
        'snow': '#FFFAFA',
        'springgreen': '#00FF7F',
        'steelblue': '#4682B4',
        'tan': '#D2B48C',
        'teal': '#008080',
        'thistle': '#D8BFD8',
        'tomato': '#FF6347',
        'turquoise': '#40E0D0',
        'violet': '#EE82EE',
        'wheat': '#F5DEB3',
        'white': '#FFFFFF',
        'whitesmoke': '#F5F5F5',
        'yellow': '#FFFF00',
        'yellowgreen': '#9ACD32'}
    colors = {i: col for i, col in enumerate(cnames.values())}
    return colors

class Str_Weak_err():
    def __init__(self, types="VDP", epi=100, train_iter=100, info=""):
        super(Str_Weak_err, self).__init__()
        self.LOSS = pd.DataFrame({
            'Types': [types],
            'Samples': [epi],
            'StrongError': ["None"],
            'WeakError': ["None"],
            'Time': [datetime.datetime.now()],
            'train_iter': [train_iter],
            'info': [info]
        })

    def CompussAndSave(self, ys, zs):
        # ys is Reference res; zs is eCLPF res;
        if len(np.shape(ys)) == 2:
            self.LOSS['StrongError'] = np.mean((zs[-1, :] - ys[-1, :]) ** 2) ** .5
            self.LOSS['WeakError'] = np.mean(np.abs(zs[-1, :]-ys[-1, :]))

        elif len(np.shape(ys)) == 3:
            self.LOSS['StrongError'] = (np.mean((zs[-1, :, 0] - ys[-1, :, 0]) ** 2 +
                                        (zs[-1, :, 1] - ys[-1, :, 1]) ** 2)) ** .5
            self.LOSS['WeakError'] = (np.mean(np.abs(zs[-1, :, 0]-ys[-1, :, 0]))
                                      + np.mean(np.abs(zs[-1, :, 1]-ys[-1, :, 1])))

        file_path = f"./Errors/{self.LOSS['Types'][0]}/"
        create_directory_if_not_exists(file_path)
        csv_filename = file_path + "ModelErrs.csv"
        file_exists = os.path.isfile(csv_filename)

        # if exist, add data
        if file_exists:
            existing_data = pd.read_csv(csv_filename, encoding='utf-8')
            # existing_data = dict(existing_data)
            Eidx = np.where(existing_data['Samples'] == self.LOSS['Samples'][0])[0]
            if len(Eidx) > 0:
                for ix in Eidx:
                    for key in existing_data:
                        vals = existing_data[key].values
                        vals[ix] = self.LOSS[key].values[0]
                        existing_data[key] = vals
                existing_data.to_csv(csv_filename, index=False)
            else:
                # Add New info.
                updated_data = pd.concat([existing_data, self.LOSS], ignore_index=True)
                updated_data.to_csv(csv_filename, index=False)
        else:
            self.LOSS.to_csv(csv_filename, index=False)

        print(f"{'='*10}")
        print(self.LOSS.to_string())
        print(f"{'=' * 10}")
        print(f"{'='*10}Success Save:{csv_filename}{'='*10}")


def MinstN(A, N):
    flat_indices = np.argsort(A.flatten())[:N]
    # 1d to 2d
    indices_2d = np.unravel_index(flat_indices, A.shape)
    return indices_2d

def create_directory_if_not_exists(directory_path):
    """
    # Example :
    # create_directory_if_not_exists("my_directory")"""
    print("=" * 40)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Creating: {directory_path}")
    else:
        print(f"Existing: {directory_path}")
    print("=" * 40)


