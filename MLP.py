import numpy as np
import pandas as pd
import keras
from keras.layers import Dense
from keras.losses import mse
from keras.utils import np_utils
from sklearn.model_selection import KFold, train_test_split
from sklearn import preprocessing
from keras.models import Sequential

df = pd.read_csv('GSE112790.csv')
df = df[['SLC25A47', 'HGFAC', 'SLC22A1', 'SLC10A1', 'CYP4F2', 'RDH16', 'LCAT', 'CYP3A4',
          'EXPH5', 'AGXT', 'AR', 'MASP2', 'ECHDC2', 'HPX', 'ABCA6', 'C8B', 'F11',
          'HRASLS2', 'ZFPM1', 'TKT', 'TXNRD1', 'SQSTM1', 'SRXN1', 'TRIM16', 'TALDO1',
          'NME1', 'ITIH1', 'FBP1', 'TFR2', 'FTCD', 'KLKB1', 'HMGCS2', 'BHMT', 'HGD',
          'GRHPR', 'KCNJ8', 'ACSM5', 'LECT2', 'PLG', 'CA5A', 'TMEM56', 'SCP2', 'CYP2J2',
          'DEPDC1B', 'MCM2', 'UBE2S', 'CDKN3', 'PTTG1', 'MAD2L1', 'GPSM2', 'HJURP',
          'MCM3', 'DEPDC1', 'GINS1', 'TRIM59', 'CEP55', 'CDCA2', 'TRIP13', 'TPX2',
          'RRM2', 'CDCA3', 'CENPO', 'MCM10', 'RACGAP1', 'NCAPG', 'FANCI', 'SHCBP1',
          'PRC1', 'CDK1', 'CDC20', 'KIF18B', 'TUBA1B', 'NEK2', 'SPC25', 'UBE2T',
          'CENPW', 'TTK', 'MTFR2', 'PBK', 'BLM', 'CDC45', 'MKI67', 'MCM5', 'ASPM',
          'KIF15', 'CENPE', 'MCM6', 'MELK', 'EXO1', 'CPN2', 'C1R', 'SERPING1',
          'SPP1', 'SSX1', 'SSX3', 'ASPSCR1', 'LAMA3', 'CDC25B', 'TUBA1C', 'CDC7',
          'DERL3', 'TMCC1', 'MAPRE3', 'PLA2G4F', 'PERM1', 'HPR', 'HAGH', 'C8A',
          'SERPINA10', 'ANXA2', 'ARPC1B', 'ITGAV', 'ARMC5', 'ISYNA1', 'PLEKHN1',
          'GTPBP2', 'MORN1', 'GNAS', 'TFRC', 'CASR', 'TRAF5', 'PDE7A', 'PLXNA1',
          'ACSL1', 'ETFDH', 'SLC1A2', 'SUSD3', 'FAM53A', 'FBLL1', 'TSC22D1',
          'SEMA4G', 'FBXO44', 'ANG', 'SERPINA4', 'BDH1', 'NPM1', 'FARSB',
          'HIC2', 'ARID3A', 'STK35', 'PROC', 'ZNF7', 'PUF60', 'RPL8', 'POLR2K',
          'C8orf33', 'DCAF13', 'C8orf76', 'VPS28', 'RPL30', 'EXOSC4', 'SCNN1A', 'MEX3A',
          'LTBP3', 'CAPS', 'LRRC74A', 'MICALL2', 'PWWP2B', 'FAAP20', 'GJC2', 'PTP4A3',
          'CELA2B', 'SRCAP', 'TECTA', 'KMT2B', 'HMGB2', 'PHF19', 'FAM83D', 'TMEM237',
          'CDK4', 'RBL1', 'POLE2', 'STMN1', 'TMEM106C', 'NUDT1', 'MSH2', 'NCAPG2', 'TK1',
          'BRCA2', 'CKAP5', 'GMPS', 'MCM7', 'MFSD2A', 'KDM8', 'AADAT', 'SERPINF2',
          'SLCO1B1', 'CYP4A11', 'AMDHD1', 'ALAD', 'HAAO', 'XDH', 'TRIM71', 'PGM1',
          'PXMP2', 'ZGPAT', 'LIME1', 'XPO1', 'ADH1B', 'CLPX', 'AOX1', 'ADH1C', 'SLC13A5',
          'C1orf162', 'BASP1', 'SIGLEC1', 'RGS12', 'CRYL1', 'HMGCL', 'PEX2', 'TERF1', 'TTPA',
          'ADH6', 'ASGR2', 'SCRIB', 'HGH1', 'CYHR1', 'GPAA1', 'FAM83H', 'ACOT12', 'ALDH8A1',
          'EVA1A', 'LPAL2', 'LPA', 'SLC39A14', 'RHOB', 'SLC7A2', 'HIPK3', 'ALAS1', 'UTP23',
          'YWHAZ', 'ATP6V1C1', 'WDYHV1', 'VPS13B', 'HLA-A', 'TAP1', 'HLA-C', 'HLA-F', 'PRDX1',
          'PABPC1', 'TMEM68', 'PRKDC', 'EXOSC10', 'LZIC', 'GAPDH', 'FAM20C', 'TAP2', 'IGF2BP3',
          'TTF2', 'SCAMP3', 'SNRPE', 'PI4KB', 'BCL6B', 'LTK', 'PLEKHO1', 'RGS10', 'CLIC1',
          'XPO5', 'VAPA', 'TNRC6A', 'C1S', 'DUSP1', 'TBC1D2B', 'SLC26A6', 'SOX6', 'KIAA1671',
          'WDR1', 'ABCC3', 'SFPQ', 'RALGAPA2', 'GALNT2', 'NEDD1', 'UBAP2L', 'GRB2', 'CLTC',
          'FGB', 'CFI', 'CDKN2C', 'COBLL1', 'BUB3', 'LBR', 'SLC20A1', 'DDIAS', 'FOXP3', 'GOT2',
          'LDHD', 'HNRNPU', 'C8orf58', 'CRYBB3', 'TPM2', 'COL4A1', 'NOTCH3', 'COL1A2', 'MAPRE1',
          'ARPC2', 'TUBB4B', 'RPL23A', 'FBL', 'RPS21', 'RFWD3', 'RPS6KC1', 'COG2', 'DESI2',
          'NUP133', 'TBCE', 'CCT7', 'HEG1', 'COL1A1', 'NRAS', 'AGFG1', 'SOCS5', 'TMEM267',
          'HSP90AB1', 'CELF1', 'KIF20B', 'NHP2', 'NIF3L1', 'NDC1', 'MAP4K4', 'TRIP4', 'GADD45B',
          'JUNB', 'SLBP', 'FAM220A', 'ZMYND19', 'NELFB', 'DNAH14', 'MPRIP', 'SPATA32', 'NUTM1',
          'ANXA10', 'RHPN1', 'NSMCE2', 'NDUFB9', 'EIF3E', 'RPL7', 'ZNF468', 'ZNF320', 'LSM2',
          'C8orf59', 'COX6C', 'EEF1D', 'PPP1R16A', 'HADH', 'NUDT6', 'ANAPC11', 'OXLD1', 'GSTZ1',
          'LTBP2', 'CDH11', 'FOS', 'NAV1', 'COL15A1', 'PDGFRA', 'HGF', 'DPF2', 'type']]


label_mapping = {label: idx for idx, label in enumerate(np.unique(df['type']))}
df['type'] = df['type'].map(label_mapping)
X = df.iloc[0:, 1:357].values
y = df.iloc[0:, -1].values
y = np_utils.to_categorical(y)
X_norm = preprocessing.normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3,
                                                    stratify=y, random_state=10)


acc_per_fold = []
loss_per_fold = []

# Merge inputs and targets
inputs = np.concatenate((X_train, X_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)

kfold = KFold(n_splits=5, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 5
for train, test in kfold.split(inputs, targets):
    model = Sequential()
    model.add(keras.Input(356))
    model.add(Dense(64, activation= 'relu'))
    model.add(Dense(32, activation= 'relu'))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    print(f'Training for fold {fold_no} ...')

    history = model.fit(inputs[train], targets[train], batch_size=16, epochs=100, verbose=1)
    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    # Increase fold number
    fold_no = fold_no + 1
    # == Provide average scores ==
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
