import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

### Laplacian filter
def laplacian_filter(senal):
  #print('forma',senal.shape)
  trials, channels, samples = senal.shape
  laplacian = np.empty((trials, samples), np.float64) 

  for trial in range(trials):
      for sample in range(samples):
        sample_sum = 0
        for channel in range(channels - 1):
            sample_sum += senal[trial, channel, sample]
        ref = senal[trial, channels - 1, sample]
        laplacian[trial, sample] = ref - (sample_sum / (channels - 1)) 
  return laplacian

### Dictionary
def dictionary_WPT(samples=1024, max_level=6, mother_wavelet='db4'):
    import pywt
    from sklearn.preprocessing import normalize
    # Base signal
    signal = np.zeros(samples)

    # Base tree
    wp_original = pywt.WaveletPacket(data=signal, wavelet=mother_wavelet, mode='periodization', maxlevel=max_level)

    dictionary = []
    dictionary_columns = []

    for level in range(1, max_level + 1):
        nodes = wp_original.get_level(level, order='natural')
        #print(f"Nivel {level}:")
        for node in nodes:
            #print(f"  Nodo: {node.path}, longitud: {len(node.data)}")
            dictionary_columns.append([node.path, len(node.data)])
            for i in range(len(node.data)):
                # New Tree
                wp_temp = pywt.WaveletPacket(data=None, wavelet=mother_wavelet, mode='periodization', maxlevel=max_level)

                # Zero Coefficients
                for other_node in wp_original.get_level(level, order='natural'):
                    wp_temp[other_node.path] = np.zeros_like(other_node.data)

                # One coefficient
                impulse = np.zeros_like(node.data)
                impulse[i] = 1
                wp_temp[node.path] = impulse

                # Rebuild signal (atom)
                signal_rec = wp_temp.reconstruct(update=False)

                # Normalize L2
                norm = np.linalg.norm(signal_rec)
                if norm > 0:
                    signal_rec /= norm

                dictionary.append(signal_rec)

    dictionary = np.array(dictionary)
    dictionary = normalize(dictionary, axis=1) 
    return dictionary_columns, dictionary

### OMP
def omp(x, D, T):
    """
    x: signal to represent (n,)
    D: dictionary (n x K)
    T: sparsity level
    Returns:
        alpha: sparse coefficient vector (K,)
    """
    n, K = D.shape
    #print(n, K, x.shape)
    r = x.copy()
    support = []
    alpha = np.zeros(n)

    for _ in range(T):
        # 1. Proyecciones
        projections = D @ r   # correlación con cada átomo

        scores = np.abs(projections)

        # 3. Evitar reselección de átomos ya elegidos
        scores[support] = -np.inf

        # 4. Elegir átomo con score máximo
        j_star = np.argmax(scores)
        support.append(j_star)

        # 5. Resolver mínimos cuadrados para soporte actual
        D_sub = D.T[:, support]
        alpha_sub = np.linalg.pinv(D_sub) @ x

        # 6. Actualizar residuo
        r = x - D_sub @ alpha_sub

    # 7. Guardar coeficientes en posiciones del soporte
    alpha[support] = alpha_sub

    return alpha, support

### AFM
def afm_function(D, support_c1, support_c2):
    K = D.shape[0]  # número de átomos
    frq_c1 = np.zeros(K)
    frq_c2 = np.zeros(K)

    # Clase 1
    for support in support_c1:          # cada support es una lista de índices activos
        frq_c1[support] += 1
    frq_c1 /= len(support_c1)           # normalizamos

    # Clase 2
    for support in support_c2:
        frq_c2[support] += 1
    frq_c2 /= len(support_c2)

    # plt.figure()
    # plt.title('Frecuencia de activación')
    # plt.plot(frq_c1, label='Clase 1')
    # plt.plot(frq_c2, label='Clase 2')
    # plt.legend()

    afm = np.zeros(K)

    for j in range(K):
        p1, p2 = frq_c1[j], frq_c2[j]
        pj_plus = max(p1, p2)
        pj_star = min(p1, p2)
        if pj_plus > 0:
            afm[j] = (pj_plus - pj_star) / pj_plus
        else:
            afm[j] = 0

    # plt.figure()
    # plt.title('Medida afm por átomo')
    # plt.stem(afm)
    # plt.show()

    return afm

### MCM
def mcm_fuction(alpha_c1, alpha_c2):
    K = alpha_c1[0].shape[0]

    # Promedio de magnitudes por átomo en cada clase
    avg_c1 = np.mean([np.abs(a) for a in alpha_c1], axis=0)
    avg_c2 = np.mean([np.abs(a) for a in alpha_c2], axis=0)

    # Medida mcm
    mcm = np.zeros(K)
    for j in range(K):
        pj1, pj2 = avg_c1[j], avg_c2[j]
        pj_plus = max(pj1, pj2)
        pj_star = min(pj1, pj2)
        if pj_plus > 0:
            mcm[j] = (pj_plus - pj_star) / pj_plus
        else:
            mcm[j] = 0

    # --- Graficar resultados ---
    # plt.figure()
    # plt.title("Promedio magnitud de coeficientes por átomo")
    # plt.plot(avg_c1, label="Clase 1")
    # plt.plot(avg_c2, label="Clase 2")
    # plt.legend()

    # plt.figure()
    # plt.title("Medida mcm por átomo")
    # plt.stem(mcm)
    # plt.show()

    return mcm

### REM
def group_by_class(X, y, num_classes):
    """
    Organiza las señales y coeficientes en listas por clase
    X: señales (n_samples x N)
    y: etiquetas (n_samples,)
    num_classes: número de clases
    """
    X_classes = [[] for _ in range(num_classes)]
    
    for x, label in zip(X, y):
        if label == 0 :    
            X_classes[0].append(x)
        else:
            X_classes[1].append(x)

    return X_classes

def rem_fast(X_classes, coeffs, D, num_classes):
    """
    Calcula la medida de error de representación discriminativa (mre) y la clase asociada a cada átomo.
    
    X_classes[c]: lista de arrays de shape (N,) de clase c
    coeffs[c]: lista de arrays de shape (K,) de coeficientes para cada señal
    D: diccionario de forma (N, K)  -> N = dimensión de señal, K = nº de átomos
    num_classes: número de clases
    """
    K, N = D.shape  
    mre = np.zeros(K)
    plus_class = np.zeros(K, dtype=int)

    # Precomputamos las reconstrucciones completas de cada clase
    recon_classes = []
    for c in range(num_classes):
        if len(X_classes[c]) == 0:
            recon_classes.append([])
            continue

        # Convertimos lista de señales y coeficientes en arrays 2D
        Xc = np.stack(X_classes[c])   # shape (num_signales_c, N)
        Ac = np.stack(coeffs[c])      # shape (num_signales_c, K)

        # Reconstrucción completa de todas las señales de la clase
        recon_c = Ac @ D
        recon_classes.append(recon_c)

    for j in range(K):
        errors = []
        for c in range(num_classes):
            if len(X_classes[c]) == 0:
                errors.append(0)
                continue

            Xc = np.stack(X_classes[c])     # (num_signales_c, N)
            Ac = np.stack(coeffs[c])        # (num_signales_c, K)
            recon_full = recon_classes[c]   # (num_signales_c, N)

            # Simula quitar átomo j
            recon_j = recon_full - np.outer(Ac[:, j], D.T[:, j])  # (num_signales_c, N)

            err_full = np.sum((Xc - recon_full)**2, axis=1)   # (num_signales_c,)
            err_drop = np.sum((Xc - recon_j)**2, axis=1)      # (num_signales_c,)

            errors.append(np.mean(err_drop - err_full))  # incremento promedio de error

        # Selección de clases
        c_plus = np.argmax(errors)
        r_plus = errors[c_plus]
        r_star = np.max([e for k, e in enumerate(errors) if k != c_plus])

        mre[j] = (r_plus - r_star) / r_plus if r_plus > 0 else 0
        plus_class[j] = c_plus

    # plt.figure()
    # plt.title("Medida mre por átomo")
    # plt.stem(mre)
    # plt.show()

    return mre, plus_class

def discriminative_score_grid(M_af, M_cm, M_rem, step=0.1):
    """
    Calcula scores discriminativos combinados usando búsqueda en cuadrícula sobre (alpha, beta).
    
    M_af, M_rem, M_cm: arrays (K,) con las medidas por átomo
    step: resolución de la cuadrícula
    
    Returns:
        best_alpha, best_beta, best_scores
    """
    K = len(M_af)
    best_scores = None
    best_alpha, best_beta = None, None
    best_eval = -np.inf   # métrica para elegir mejor combinación (puede ser np.max(scores), etc.)
    
    # Barrido sobre alpha y beta
    for alpha in np.arange(0, 1+step, step):
        for beta in np.arange(0, 1+step, step):
            if alpha + beta <= 1:  # restricción
                scores = alpha*M_af + beta*M_rem + (1 - alpha - beta)*M_cm
                
                # Ejemplo de criterio: maximizamos la dispersión entre átomos
                eval_metric = np.var(scores)  # podría usarse otra métrica del paper
                
                if eval_metric > best_eval:
                    best_eval = eval_metric
                    best_scores = scores
                    best_alpha, best_beta = alpha, beta
    
    return best_alpha, best_beta, best_scores

### Dictionay per class
def sub_dictionary(plus_class, best_scores, n_atoms_per_class = 20, num_classes = 2):
    subdicts = []

    for c in range(num_classes):
        idx_class_c = np.where(plus_class == c)[0]             # átomos de la clase
        idx_sorted = idx_class_c[np.argsort(-best_scores[idx_class_c])]  # más discriminativos primero
        selected_idx = idx_sorted[:n_atoms_per_class]
        subdicts.append(D.T[:, selected_idx])

    final_dict = np.hstack(subdicts)  # diccionario final discriminativo
    #print("Shape diccionario final:", final_dict.shape)
    
    return final_dict


################# MAIN #####################################

dictionary = np.load('dictionary_512.npy', allow_pickle=True).item()
print(type(dictionary))
volunteers = sorted(dictionary.keys())
print('Volunteers:',volunteers)

df_info = pd.read_excel('C:/Users/Usuario/Documents/GitHub/Data_base_processing/Database_Info.xlsx')
upper_dominant = (df_info["Upper Dominant Laterality"]).tolist()
lower_dominant = (df_info["Lower Dominant Laterality"]).tolist()
print('Upper dominant=',upper_dominant)
print('Lower dominant=',lower_dominant)

feet_channels = ['CZ', 'C4', 'C3', 'FZ', 'PZ']

trials, channels, samples = dictionary['S01']['Lower']['Task1']['X']['Rest'].shape
print(f"Trials={trials}|Channels={channels}|Samples={samples}")

## Create Dictionary
coef_dictionary, D = dictionary_WPT()
print('Shape dictionary', D.shape)

### Analysis per Volunteer
acc_vol = []
for volunteer in range(1,31):

    if volunteer < 10:
        ini = 'S0'
    else:
        ini = 'S'

    acc = []
    tpr = []

    X = dictionary[ini+str(volunteer)]['Lower']['Task1']['X']
    Y = dictionary[ini+str(volunteer)]['Lower']['Task1']['Y']

    laplace_rest = laplacian_filter(X['Rest'])

    if lower_dominant[volunteer-1] == 'Right':
        laplace_right = laplacian_filter(X['Right'])
        index = np.random.choice(laplace_rest.shape[0], 40, replace=False) 
        selection = laplace_rest[index]
        X = np.concatenate((laplace_right, selection), axis=0)
    else:
        laplace_left = laplacian_filter(X['Left'])
        index = np.random.choice(laplace_rest.shape[0], 40, replace=False) 
        selection = laplace_rest[index]
        X = np.concatenate((laplace_left, selection), axis=0)
    
    y = np.concatenate((np.ones(40),np.zeros(40)), axis=0)

    #print('Tamaño X:', X.shape, 'Tamaño y:', y.shape)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold CV
    acc_cv = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]


        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        #print('X_train size:',X_train.shape,'X_test size:',X_test.shape,'y_train size:',len(y_test),'y_test size:',len(y_train))
        #print('Dictionary size:',D.shape)

        T = 50     # nivel de dispersión

        alpha_c1 = []
        alpha_c2 = []
        support_c1 = []
        support_c2 = []

        # Alpha y soporte de cada señal por clase
        for trial,clase in zip(X_train,y_train):
            alpha_temp, support_temp = omp(trial, D, T)
            if clase == 0:
                alpha_c1.append(alpha_temp)
                support_c1.append(support_temp)
            else:
                alpha_c2.append(alpha_temp)
                support_c2.append(support_temp)

        ##AFM
        afm = afm_function(D, support_c1, support_c2)

        ##MCM
        mcm = mcm_fuction(alpha_c1, alpha_c2)

        ##REM
        X_classes = group_by_class(X_train, y_train, 2)
        mre, plus_class = rem_fast(X_classes, [alpha_c1, alpha_c2], D, 2)

        ###Discriminative score
        best_alpha, best_beta, best_scores = discriminative_score_grid(afm, mcm, mre)

        ###Create subdictionaries
        #final_dict = sub_dictionary(plus_class, best_scores, n_atoms_per_class = 20, num_classes = 2)

        ###Data proyections
        #X_train_proj = X_train @ final_dict
        #X_test_proj = X_test @ final_dict

        ###Classificator
        acc_atom = []
        num_athoms = 100
        for atom in range(1,num_athoms):

            final_dict = sub_dictionary(plus_class, best_scores, n_atoms_per_class = atom, num_classes = 2)

            X_train_proj = X_train @ final_dict
            X_test_proj = X_test @ final_dict

            lda = LinearDiscriminantAnalysis()
            lda.fit(X_train_proj, y_train)
            y_pred = lda.predict(X_test_proj)
            acc = accuracy_score(y_test, y_pred)
            acc_atom.append(acc)

        # plt.figure()
        # plt.plot(range(1,num_athoms),acc_atom)
        # plt.title("Acuraccy per athom")
        # plt.xlabel('athoms')
        # plt.ylabel('Accuracy')
        # plt.show()
        acc_cv.append(acc_atom)

    # Promedio sobre folds
    acc_cv = np.array(acc_cv)  # shape = (n_folds, num_atoms-1)
    mean_acc = np.mean(acc_cv, axis=0)
    std_acc = np.std(acc_cv, axis=0)

    plt.figure()
    plt.plot(range(1,100), mean_acc, label="Mean CV accuracy")
    plt.fill_between(range(1,100), mean_acc-std_acc, mean_acc+std_acc, alpha=0.2)
    plt.xlabel("Número de átomos")
    plt.ylabel("Accuracy")
    plt.title(f"Voluntario {volunteer} - 5-fold CV")
    plt.legend()
    plt.show()
    acc_vol.append(mean_acc)


plt.figure()
for i in acc_vol:
    plt.plot(i)
plt.title("Acuraccy per athom")
plt.xlabel('athoms')
plt.ylabel('Accuracy')
plt.show()





