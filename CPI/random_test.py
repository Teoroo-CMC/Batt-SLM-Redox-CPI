import os
import sys
import numpy as np
from pathlib import Path

CPI_DIR = Path("your_own_path/Batt-SLM-Redox-CPI/CPI")

# directory of the intermediate files
INPUT_DIR = CPI_DIR / "Input"
if not os.path.exists(INPUT_DIR):
    os.mkdir(INPUT_DIR)

# for BOLTZMANN average
BOLTZMANN_CONSTANT = 0.0019872041  # k, kcal/mol/K
TEMPERATURE = 298.15  # temperature, (K)

# CDE: class 1; WSE: class 0
def load_data():

    # get features of molecular
    mol_id = []
    raw_features = []
    raw_data = []
    raw_SMILES = []
    raw_Category = []
    raw_label = []
    with open(INPUT_DIR / "Features-NoF.csv") as g:

        lines = g.readlines()
        titles = lines[0].strip("\n").split(";")

        id_col = titles.index("Abbreviation")
        smiles_col = titles.index("SMILES")
        category_col = titles.index("Category")
        function_col = titles.index("Function")
        sasa_col = titles.index("SASA(O/N,max)")

        raw_features = titles[sasa_col:]
        for mol_index in range(0, len(lines[1:])):
            line_index = mol_index + 1
            items = lines[line_index].strip("\n").replace(",", ".").split(";")
            mol_id.append(items[id_col].split("(")[0])
            raw_SMILES.append(items[smiles_col])
            raw_Category.append(items[category_col])
            raw_data.append(items[sasa_col:])
            function_class = 1 # related to the solvating power, so set label = 1 for CDE
            if "WSE" in items[function_col]:
                function_class = 0
            raw_label.append(function_class)

    raw_features = np.array(raw_features)
    raw_data = np.array(raw_data).astype(np.float64)
    raw_SMILES = np.array(raw_SMILES)
    raw_Category = np.array(raw_Category)
    raw_label = np.array(raw_label).astype(np.float64)
    
    return raw_features,raw_SMILES,raw_Category,raw_data,raw_label
    

# CDE: class 1; WSE/Diluent: class 0
def train_logistic_classifer(random_seed, raw_features, data_train, label_train):

    model_name = "RandomTest"

    model_dir = CPI_DIR / model_name
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # change the standard output file
    result_file = model_dir / f"result-{random_seed}.txt"
    result_file = open(result_file, "w")
    old_std_out = sys.stdout
    sys.stdout = result_file

    # normalized by the averaged values
    print("Features =", raw_features)
    feature_mean = np.mean(data_train, axis=0)
    feature_mean = np.where(feature_mean == 0, 1e-6, feature_mean)
    print("Feature mean values before normalized = ", feature_mean)
    data_train = data_train / feature_mean
    feature_mean = np.mean(data_train, axis=0)
    print("Feature mean values after normalized = ", feature_mean)

    # use 0.5 as the binary classification point
    print("Num of WSE = ", len(np.where(label_train < 0.5)[0]))
    print("Num of CDE = ", len(np.where(label_train > 0.5)[0]))

    # constrain the coefficient > 0.0
    from scipy.optimize import minimize
    from sklearn.metrics import log_loss # Cross-Entropy Loss
    from sklearn.preprocessing import add_dummy_feature

    # to deal with the extremely large/negative values for original sigmoid
    # def sigmoid(z):
    #     result = np.zeros_like(z)
    #     mask = z >= 0
    #     result[mask] = 1 / (1 + np.exp(-z[mask]))
    #     mask = z < 0
    #     result[mask] = np.exp(z[mask]) / (1 + np.exp(z[mask]))
    #     return result

    def logistic_loss(params, X, y):
        w = params  # params: intercept and coef
        z = X @ w  # linear combination : z = w0 + w1*x1 + ... + wn*xn
        y_pred = 1 / (1 + np.exp(-z))  # Predicted probabilities p(y=1|x), original Sigmoid
        margin_penalty = 30.0 * np.mean(np.maximum(0, 1 - y * z)) # maximal the margin, see SVM
        return log_loss(y, y_pred)+margin_penalty

    train_data_bias = add_dummy_feature(data_train) # add bias before the first column

    # the coefficients should be positive
    # fitting on whole dataset, because of the size of dataset is small and we expected the CPI can completely seperate the current WSE and CDEs
    init_params = [1.0,1.0,1.0,1.0,1.0]
    constraints = [{'type': 'ineq', 'fun': lambda params: params[i]} for i in range(1, train_data_bias.shape[1])]
    classifier = minimize(fun=logistic_loss, x0=init_params, args=(train_data_bias, label_train),
                       constraints=constraints, method='SLSQP')
    intercept = classifier.x[0]
    coeffs = classifier.x[1:]

    # modify the coeff and intercept to scale the predictions within 0 - 1
    formula = ""
    for index in range(0, len(raw_features)):
        formula += "(%.3f)*(%s)+"%(coeffs[index], raw_features[index])
    formula += "(%.3f)"%(intercept)

    origin_pred = []
    for mol_index in range(0, len(label_train)):
        temp_formula = formula
        for feature_index in range(0, len(raw_features)):
            temp_formula = temp_formula.replace(f"({raw_features[feature_index]})",
                                                  f"({data_train[mol_index][feature_index]})")
        origin_pred.append(eval(temp_formula))

    # shift to 0
    origin_pred = np.array(origin_pred)
    intercept = intercept - np.min(origin_pred)
    origin_pred = origin_pred - np.min(origin_pred)
    # scale to 0-1
    intercept /= np.max(origin_pred)
    coeffs /= np.max(origin_pred)

    print("-------------------coefficient and intercept_----------------")
    final_formula = ""
    for index in range(0, len(raw_features)):
        final_formula += "(%.3f)*(%s)+"%(coeffs[index], raw_features[index])
    final_formula += "(%.3f)"%(intercept)
    print(final_formula)

    # reset std out
    sys.stdout.flush()
    sys.stdout = old_std_out
    result_file.close()

#
def predict_cpi(random_seed, raw_features, data, labels, data_type):
    
    model_name = "RandomTest"
    model_dir = CPI_DIR / model_name
    if not os.path.exists(model_dir):
        raise ValueError(f"No models {model_name} were found")
    
    feature_mean = []
    result_file = model_dir / f"result-{random_seed}.txt"
    with open(result_file, "r") as f:
        lines = f.readlines()
        assert "Feature mean values before normalized =" in lines[1]
        feature_mean = lines[1].split("=")[1].strip(" []\n").split()
        feature_mean = np.array(feature_mean, dtype=np.float32)

    data_scaled = data / feature_mean
    feature_mean = np.mean(data_scaled, axis=0)

    final_formula = ""
    with open(result_file, "r") as f:
        lines = f.readlines()
        is_formula_start = False
        for line in lines:
            if "coefficient and intercept" in line:
                is_formula_start = True
                continue
            if is_formula_start:
                final_formula = line.strip(" \n+")
                is_formula_start = False
                break

    # predict
    predict_labels = [0] * len(labels)
    for mol_index in range(0, len(labels)):
        temp_formula = final_formula
        for feature_index in range(0, len(raw_features)):
            temp_formula = temp_formula.replace(f"({raw_features[feature_index]})", f"({data_scaled[mol_index][feature_index]})")
        predict = eval(temp_formula)
        if predict > 0.5:
            predict_labels[mol_index] = 1

    # output
    append_result_file = open(result_file, "a")
    old_std_out = sys.stdout
    sys.stdout = append_result_file
    from sklearn.metrics import classification_report
    print(f"--------------------{data_type}------------------")
    print(classification_report(labels, predict_labels))
    sys.stdout.flush()
    sys.stdout = old_std_out
    append_result_file.close()

    report = classification_report(labels, predict_labels, output_dict=True)
    return report,np.array(predict_labels)
    
# 
if __name__ == "__main__":

    from collections import Counter
    from sklearn.model_selection import train_test_split

    raw_features,raw_SMILES,raw_Category,raw_data,raw_label = load_data()
    counter = Counter(raw_Category)

    train_result_class_0 = []
    train_result_class_1 = []
    valid_result_class_0 = []
    valid_result_class_1 = []
    metric = ["precision", "recall", "f1-score", "support"]
    for random_seed in [2,3,4]:

        # if only one samples with a specific category, put it into the training set
        indices = np.arange(len(raw_Category))
        rare_idx = [i for i, c in enumerate(raw_Category) if counter[c] < 2]

        # other, stratify class
        stratify_idx = [i for i, c in enumerate(raw_Category) if counter[c] >= 2]
        train_idx, valid_idx = train_test_split(stratify_idx, test_size=0.2, random_state=random_seed,
                                                stratify=[raw_Category[i] for i in stratify_idx])
        train_idx = np.concatenate([train_idx, rare_idx])

        # training 
        data_train = raw_data[train_idx]
        data_valid = raw_data[valid_idx]
        label_train = raw_label[train_idx]
        label_valid = raw_label[valid_idx]
        train_logistic_classifer(random_seed, raw_features, data_train, label_train)

        # predict
        train_report,predict_train_labels = predict_cpi(random_seed, raw_features, data_train, label_train, "train")
        train_result_class_0.append([train_report["0.0"][key] for key in metric])
        train_result_class_1.append([train_report["1.0"][key] for key in metric])
        valid_report,predict_valid_labels = predict_cpi(random_seed, raw_features, data_valid, label_valid, "valid")
        valid_result_class_0.append([valid_report["0.0"][key] for key in metric])
        valid_result_class_1.append([valid_report["1.0"][key] for key in metric])

        # print the true and predict label on validation set
        smiles_train = raw_SMILES[train_idx]
        category_train = raw_Category[train_idx]
        train_csv_file = CPI_DIR / "RandomTest" / f"train_{random_seed}_results.csv"
        with open(train_csv_file, "w") as g:
            g.write("SMILES;Category;Label;Predict\n")
            for v_i in range(0, len(smiles_train)):
                g.write("%s;%s;%f;%f\n"%(smiles_train[v_i], category_train[v_i], label_train[v_i], predict_train_labels[v_i]))

        # print the true and predict label on validation set
        smiles_valid = raw_SMILES[valid_idx]
        category_valid = raw_Category[valid_idx]
        valid_csv_file = CPI_DIR / "RandomTest" / f"valid_{random_seed}_results.csv"
        with open(valid_csv_file, "w") as g:
            g.write("SMILES;Category;Label;Predict\n")
            for v_i in range(0, len(smiles_valid)):
                g.write("%s;%s;%f;%f\n"%(smiles_valid[v_i], category_valid[v_i], label_valid[v_i], predict_valid_labels[v_i]))

    # output final results
    model_dir = CPI_DIR / "RandomTest"
    result_file = model_dir / "final.txt"
    final_result_file = open(result_file, "w")
    old_std_out = sys.stdout
    sys.stdout = final_result_file

    print("--------------------------Train results  -----------------------")
    print("class    precision    recall    f1-score    support")
    results_mean = np.mean(train_result_class_0, axis=0)
    results_std = np.std(train_result_class_0, axis=0)
    print("0    %.3f±%.3f    %.3f±%.3f    %.3f±%.3f    %d"%(results_mean[0], results_std[0],
        results_mean[1], results_std[1], results_mean[2], results_std[2], results_mean[3]))
    results_mean = np.mean(train_result_class_1, axis=0)
    results_std = np.std(train_result_class_1, axis=0)
    print("1    %.3f±%.3f    %.3f±%.3f    %.3f±%.3f    %d"%(results_mean[0], results_std[0],
        results_mean[1], results_std[1], results_mean[2], results_std[2], results_mean[3]))

    print("--------------------------Test results  ---------------------")
    print("class    precision    recall    f1-score    support")
    results_mean = np.mean(valid_result_class_0, axis=0)
    results_std = np.std(valid_result_class_0, axis=0)
    print("0    %.3f ± %.3f    %.3f ± %.3f    %.3f ± %.3f    %d"%(results_mean[0], results_std[0],
        results_mean[1], results_std[1], results_mean[2], results_std[2], results_mean[3]))
    results_mean = np.mean(valid_result_class_1, axis=0)
    results_std = np.std(valid_result_class_1, axis=0)
    print("1    %.3f ± %.3f    %.3f ± %.3f    %.3f ± %.3f    %d"%(results_mean[0], results_std[0],
        results_mean[1], results_std[1], results_mean[2], results_std[2], results_mean[3]))
    
    sys.stdout.flush()
    sys.stdout = old_std_out
    final_result_file.close()
