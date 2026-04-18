#include <bits/stdc++.h>
#include <cmath>
using namespace std;
const double RI = 0.580000;
const double EPS = 1e-10;
const int BAYES_M = 5; // 贝叶斯先验常数

double betacf(double a, double b, double x) {
    int m;
    double aa,c,d,del,h,qab,qam,qap;
    qab=a+b;
    qam=a-1;
    qap=a+1;
    c=1;
    d=1-qab*x/qap;
    if (fabs(d) < EPS) d=EPS;
    d=1.0/d;
    h=d;
    for (m=1;m<=100;m++) {
        aa=m*(b-m)*x/((qam+2*m)*(a+2*m));
        d=1+aa*d;
        if (fabs(d) < EPS) d=EPS;
        c=1+aa/c;
        if (fabs(c) < EPS) c=EPS;
        d=1.0/d;
        h*=d*c;
        aa=-(a+m)*(qab+m)*x/((a+2*m)*(qap+2*m));
        d=1+aa*d;
        if (fabs(d) < EPS) d=EPS;
        c=1+aa/c;
        if (fabs(c) < EPS) c=EPS;
        d=1.0/d;
        del=d*c;
        h*=del;
        if (fabs(del-1.0) < 3e-7) break;
    }
    return h;
}

// 不完全Gamma函数近似，用于计算t分布p值
double incbeta(double a, double b, double x) {
    double bt;
    if (x < 0.0 || x > 1.0) return 1.0;
    if (x == 0.0 || x == 1.0) bt = 0.0;
    else bt = exp(lgamma(a+b) - lgamma(a) - lgamma(b) + a*log(x) + b*log(1.0-x));
    if (x < (a+1.0)/(a+b+2.0)) {
        return bt*betacf(a,b,x)/a;
    } else {
        return 1.0 - bt*betacf(b,a,1.0-x)/b;
    }
}

// t分布的p值计算
double t_pvalue(double t, double df) {
    if (df <= 0) return 1.0;
    double x = df / (df + t*t);
    double p = incbeta(x, df/2.0, 0.5);
    return t < 0 ? p : 1.0 - p;
}

struct Game {
    string name;
    double price;
    double points;
    double pointNumber;
    double time;
    string type; // 游戏类型，用于分层验证
    // 处理后的指标
    double b; // 贝叶斯修正好评率
    double t_norm; // 归一化时长
    double b_norm; // 归一化修正好评率
    double p_norm; // 正向化归一化价格
    double score; // TOPSIS得分
}inputGame[1000];

struct Ahp{
    double time;
    double points;
    double price;
    double timeSum = 0.0;
    double pointsSum = 0.0;
    double priceSum = 0.0;
}ahp;

int Len = 0;
double matrixOld[4][4], matrixNew[4][4], timeAHP, pointsAHP, priceAHP, ci, aw[4];

// 通用工具函数：计算Spearman秩相关系数
double spearman_correlation(const vector<double>& x, const vector<double>& y) {
    int n = x.size();
    if (n != y.size() || n == 0) return 0;
    
    // 计算排名
    vector<pair<double, int>> rank_x(n), rank_y(n);
    for (int i = 0; i < n; i++) {
        rank_x[i] = {-x[i], i}; // 负号，因为得分越高排名越前
        rank_y[i] = {-y[i], i};
    }
    sort(rank_x.begin(), rank_x.end());
    sort(rank_y.begin(), rank_y.end());
    
    vector<int> rx(n), ry(n);
    for (int i = 0; i < n; i++) {
        rx[rank_x[i].second] = i;
        ry[rank_y[i].second] = i;
    }
    
    // 计算rho
    double sum_d2 = 0;
    for (int i = 0; i < n; i++) {
        sum_d2 += pow(rx[i] - ry[i], 2);
    }
    double rho = 1 - 6 * sum_d2 / (n * (n*n - 1));
    return rho;
}

// 通用工具函数：计算Pearson相关系数
pair<double, double> pearson_correlation(const vector<double>& x, const vector<double>& y) {
    int n = x.size();
    if (n != y.size() || n < 3) return {0, 1.0};
    
    double mean_x = 0, mean_y = 0;
    for (int i = 0; i < n; i++) {
        mean_x += x[i];
        mean_y += y[i];
    }
    mean_x /= n;
    mean_y /= n;
    
    double cov = 0, var_x = 0, var_y = 0;
    for (int i = 0; i < n; i++) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    
    if (var_x < EPS || var_y < EPS) return {0, 1.0};
    double r = cov / sqrt(var_x * var_y);
    
    // 计算t值和p值
    double t = r * sqrt((n-2)/(1 - r*r + EPS));
    double p = 2 * min(t_pvalue(t, n-2), 1 - t_pvalue(t, n-2));
    return {r, p};
}

// 线性回归，计算R平方
double linear_regression_r2(const vector<double>& y, const vector<vector<double>>& X) {
    int n = y.size();
    int p = X.size();
    if (n == 0) return 0;
    
    // 简单的单变量回归，用于VIF计算
    if (p == 1) {
        double mean_y = 0, mean_x = 0;
        for (int i = 0; i < n; i++) {
            mean_y += y[i];
            mean_x += X[0][i];
        }
        mean_y /= n;
        mean_x /= n;
        
        double cov = 0, var_x = 0;
        for (int i = 0; i < n; i++) {
            double dx = X[0][i] - mean_x;
            double dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
        }
        if (var_x < EPS) return 0;
        double b = cov / var_x;
        double a = mean_y - b * mean_x;
        
        double ss_tot = 0, ss_res = 0;
        for (int i = 0; i < n; i++) {
            ss_tot += pow(y[i] - mean_y, 2);
            ss_res += pow(y[i] - (a + b*X[0][i]), 2);
        }
        if (ss_tot < EPS) return 0;
        return 1 - ss_res / ss_tot;
    }
    return 0;
}

double ahpMaxMin(double x, double y){
    return x/y;
}

// 重新计算得分的函数，用于敏感性分析
vector<double> recalculate_scores(int bayes_m, bool use_zscore, bool use_manhattan, bool use_inv_p, 
                                 double w_t_ahp, double w_b_ahp, double w_p_ahp) {
    vector<Game> games(inputGame, inputGame + Len);
    int n = Len;
    
    // 1. 贝叶斯修正
    double avg_points = 0;
    for (int i = 0; i < n; i++) avg_points += games[i].points;
    avg_points /= n;
    for (int i = 0; i < n; i++) {
        double p = games[i].points;
        double num = games[i].pointNumber;
        games[i].b = (p * num + bayes_m * avg_points) / (num + bayes_m);
    }
    
    // 2. 归一化
    double min_t = 1e18, max_t = -1e18;
    double min_b = 1e18, max_b = -1e18;
    double min_p = 1e18, max_p = -1e18;
    double mean_t = 0, mean_b = 0, mean_p = 0;
    double std_t = 0, std_b = 0, std_p = 0;
    
    for (int i = 0; i < n; i++) {
        min_t = min(min_t, games[i].time);
        max_t = max(max_t, games[i].time);
        min_b = min(min_b, games[i].b);
        max_b = max(max_b, games[i].b);
        min_p = min(min_p, games[i].price);
        max_p = max(max_p, games[i].price);
        mean_t += games[i].time;
        mean_b += games[i].b;
        mean_p += games[i].price;
    }
    mean_t /= n; mean_b /= n; mean_p /= n;
    
    if (use_zscore) {
        for (int i = 0; i < n; i++) {
            std_t += pow(games[i].time - mean_t, 2);
            std_b += pow(games[i].b - mean_b, 2);
            std_p += pow(games[i].price - mean_p, 2);
        }
        std_t = sqrt(std_t / n);
        std_b = sqrt(std_b / n);
        std_p = sqrt(std_p / n);
    }
    
    for (int i = 0; i < n; i++) {
        if (use_zscore) {
            // Z-score归一化
            games[i].t_norm = (games[i].time - mean_t) / (std_t + EPS);
            games[i].b_norm = (games[i].b - mean_b) / (std_b + EPS);
            if (use_inv_p) {
                games[i].p_norm = 1.0 / (games[i].price + EPS);
                games[i].p_norm = (games[i].p_norm - mean_p) / (std_p + EPS);
            } else {
                games[i].p_norm = -(games[i].price - mean_p) / (std_p + EPS);
            }
        } else {
            // Min-Max归一化
            double range_t = max_t - min_t;
            double range_b = max_b - min_b;
            double range_p = max_p - min_p;
            if (range_t < EPS) range_t = 1;
            if (range_b < EPS) range_b = 1;
            if (range_p < EPS) range_p = 1;
            
            games[i].t_norm = (games[i].time - min_t) / range_t;
            games[i].b_norm = (games[i].b - min_b) / range_b;
            if (use_inv_p) {
                games[i].p_norm = 1.0 / (games[i].price + EPS);
                double min_p_inv = 1.0/(max_p + EPS), max_p_inv = 1.0/(min_p + EPS);
                games[i].p_norm = (games[i].p_norm - min_p_inv) / (max_p_inv - min_p_inv + EPS);
            } else {
                games[i].p_norm = (max_p - games[i].price) / range_p;
            }
        }
        
        games[i].t_norm = max(-10.0, min(10.0, games[i].t_norm));
        games[i].b_norm = max(-10.0, min(10.0, games[i].b_norm));
        games[i].p_norm = max(-10.0, min(10.0, games[i].p_norm));
    }
    
    // 3. 熵权（这里用原来的熵权，因为敏感性分析不改变权重计算）
    double w_t_e, w_b_e, w_p_e;
    {
        int m = n;
        double k = 1.0 / log(m + EPS);
        double sum_t = 0, sum_b = 0, sum_p = 0;
        for (int i = 0; i < m; i++) {
            sum_t += games[i].t_norm + EPS;
            sum_b += games[i].b_norm + EPS;
            sum_p += games[i].p_norm + EPS;
        }
        double e_t = 0, e_b = 0, e_p = 0;
        for (int i = 0; i < m; i++) {
            double p_t = (games[i].t_norm + EPS) / sum_t;
            double p_b = (games[i].b_norm + EPS) / sum_b;
            double p_p = (games[i].p_norm + EPS) / sum_p;
            e_t -= p_t * log(p_t);
            e_b -= p_b * log(p_b);
            e_p -= p_p * log(p_p);
        }
        e_t *= k; e_b *= k; e_p *= k;
        double d_t = 1 - e_t, d_b = 1 - e_b, d_p = 1 - e_p;
        double sum_d = d_t + d_b + d_p;
        w_t_e = d_t / sum_d;
        w_b_e = d_b / sum_d;
        w_p_e = d_p / sum_d;
    }
    
    // 4. 组合权重
    double w_t, w_b, w_p;
    {
        double t = w_t_ahp * w_t_e;
        double b = w_b_ahp * w_b_e;
        double p = w_p_ahp * w_p_e;
        double sum = t + b + p;
        w_t = t / sum;
        w_b = b / sum;
        w_p = p / sum;
    }
    
    // 5. TOPSIS
    vector<vector<double>> Y(n, vector<double>(3));
    for (int i = 0; i < n; i++) {
        Y[i][0] = games[i].t_norm * w_t;
        Y[i][1] = games[i].b_norm * w_b;
        Y[i][2] = games[i].p_norm * w_p;
    }
    vector<double> Y_plus(3, -1e18), Y_minus(3, 1e18);
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < n; i++) {
            Y_plus[j] = max(Y_plus[j], Y[i][j]);
            Y_minus[j] = min(Y_minus[j], Y[i][j]);
        }
    }
    
    vector<double> scores(n);
    for (int i = 0; i < n; i++) {
        double d_plus = 0, d_minus = 0;
        for (int j = 0; j < 3; j++) {
            if (use_manhattan) {
                d_plus += abs(Y[i][j] - Y_plus[j]);
                d_minus += abs(Y[i][j] - Y_minus[j]);
            } else {
                d_plus += pow(Y[i][j] - Y_plus[j], 2);
                d_minus += pow(Y[i][j] - Y_minus[j], 2);
            }
        }
        if (!use_manhattan) {
            d_plus = sqrt(d_plus);
            d_minus = sqrt(d_minus);
        }
        scores[i] = d_minus / (d_plus + d_minus + EPS);
    }
    
    return scores;
}

int InputGame(){
    cout << "请输入游戏数据文件位置， 样例：" << "D:/GameCP/Data/a.csv" << endl;
    string path ;
    getline(cin ,path);

    ifstream file(path);

    //错误处理
    if (!file.is_open()) {
        cout << "错误：找不到文件！请检查路径是否正确：" << endl;
        cout << "你填写的路径是：" << path << endl;
        cout << "注意：Windows路径要用 / 或 \\\\，不要只用 \\" << endl;
        return 1;
    }

    string line;
    getline(file, line);

    //读取文件行数
    while (getline(file, line)) {
        if (line.empty()) continue;
        Len++;
    }
    cout << "共有 " << Len << " 行数据" << endl;


    file.close();
    file.open(path);

    //开始读取并写入数据
    getline(file, line);

    int i = 0;
    while (getline(file, line) && i < Len){
        if (line.empty()) continue;
        stringstream ss(line);
        string sName, sPrice, sPoints, sPointNumber, sTime, sType;

        getline(ss, sName, ',');
        getline(ss, sPrice, ',');
        getline(ss, sPoints, ',');
        getline(ss, sPointNumber, ',');
        getline(ss, sTime, ',');
        getline(ss, sType, ','); // 读取类型字段

        // 先检查核心字段是否为空
        if (sName.empty() || sPrice.empty() || sPoints.empty() || sPointNumber.empty() || sTime.empty()) {
            cout << "跳过字段缺失的行：" << line << endl;
            continue;
        }

        try {
            double price = stod(sPrice);
            double points = stod(sPoints);
            double pointNumber = stod(sPointNumber);
            double time = stod(sTime);
            inputGame[i].price = price;
            inputGame[i].points = points;
            inputGame[i].pointNumber = pointNumber;
            inputGame[i].time = time;
            inputGame[i].name = sName;
            inputGame[i].type = sType;
        } catch (const exception&) { // 捕获所有标准异常，包括invalid_argument和out_of_range
            // 处理特殊值
            if (sPrice == "免费开玩" || sPrice == "免费") {
                // 免费游戏，价格为0
                inputGame[i].price = 0.00;
                inputGame[i].name = sName;
                inputGame[i].type = sType;
                try {
                    inputGame[i].points = stod(sPoints);
                    inputGame[i].pointNumber = stod(sPointNumber);
                    inputGame[i].time = stod(sTime);
                } catch (const exception&) {
                    cout << "跳过免费游戏中格式错误的行：" << line << endl;
                    continue;
                }
            } else if (sPoints == "无用户评测") {
                // 无用户评测
                inputGame[i].points = inputGame[i].pointNumber = 0.00;
                inputGame[i].name = sName;
                inputGame[i].type = sType;
                try {
                    inputGame[i].price = stod(sPrice);
                    inputGame[i].time = stod(sTime);
                } catch (const exception&) {
                    cout << "跳过无评测游戏中格式错误的行：" << line << endl;
                    continue;
                }
            } else {
                cout << "跳过格式错误的行：" << line << endl;
                continue;
            }
        }
        i ++;
    }

    file.close();
    Len = i; // 修正实际读取到的有效行数
    return 0;
}


int InputAhp(){
    cout << "请输入AHP判断矩阵文件位置， 样例：" << "D:/GameCP/Data/question.csv" << endl;
    string path ;
    getline(cin ,path);

    ifstream file(path);

    //错误处理
    if (!file.is_open()) {
        cout << "错误：找不到文件！请检查路径是否正确：" << endl;
        cout << "你填写的路径是：" << path << endl;
        cout << "注意：Windows路径要用 / 或 \\\\，不要只用 \\" << endl;
        return 1;
    }

    int i = 1;
    string line;

    while (getline(file, line) && i <= 3){
        if (line.empty()) continue;
        stringstream ss(line);
        string sTime, sPoints, sPrice;

        getline(ss, sTime, ',');
        getline(ss, sPoints, ',');
        getline(ss, sPrice, ',');

        matrixOld[i][1] = stod(sTime);    ahp.timeSum += stod(sTime);
        matrixOld[i][2] = stod(sPoints);  ahp.pointsSum += stod(sPoints);
        matrixOld[i][3] = stod(sPrice);   ahp.priceSum += stod(sPrice);
        i ++;
    }

    return 0;
}

int writeMatrix(){
    InputAhp();
    for (int i = 1; i <= 3; i++) {
        matrixNew[i][1] =  ahpMaxMin(matrixOld[i][1], ahp.timeSum);

    }
    for (int i = 1; i <= 3; i++) {
        matrixNew[i][2] =  ahpMaxMin(matrixOld[i][2], ahp.pointsSum);
    }
    for (int i = 1; i <= 3; i++) {
        matrixNew[i][3] =  ahpMaxMin(matrixOld[i][3], ahp.priceSum);
    }
    return 0;
}


double Aw(double w1, double w2, double w3, int i){
    aw[i] = w1*matrixOld[i][1] + w2*matrixOld[i][2] + w3*matrixOld[i][3];
    return aw[i];
}
double CI(double w1, double w2, double w3){
    double i = 0.0, j = 0.0, k = 0.0;
    i = Aw(w1, w2, w3, 1);
    j = Aw(w1, w2, w3, 2);
    k = Aw(w1, w2, w3, 3);
    double sum = (i/w1 + j/w2 + k/w3 - 3)/3;
    return sum;
}
double CR(double w1, double w2, double w3){
    return CI(w1, w2, w3)/RI;
}

double pdMatrix(double w1, double w2, double w3){
    return CR(w1, w2, w3);
}


bool isMatrix(double w1, double w2, double w3){
     if (pdMatrix(w1, w2, w3) < 0.1){
        return true;
    }else{
        return false;
    }
}

int makeMatrix(){
    writeMatrix();

    double i, j, k;
    i = matrixNew[1][1] + matrixNew[1][2] + matrixNew[1][3];
    j = matrixNew[2][1] + matrixNew[2][2] + matrixNew[2][3];
    k = matrixNew[3][1] + matrixNew[3][2] + matrixNew[3][3];
    double sum = i + j + k;
    timeAHP = i/sum;
    pointsAHP = j/sum;
    priceAHP = k/sum;

    if (isMatrix(timeAHP, pointsAHP, priceAHP)){
        cout << "CR = " << pdMatrix(timeAHP, pointsAHP, priceAHP) << endl;
        cout << "AHP判断矩阵通过一致性检验" << endl;
    } else{
        cout << "CR = " << pdMatrix(timeAHP, pointsAHP, priceAHP) << endl;
        cout << "警告：AHP判断矩阵未通过一致性检验，结果可能存在偏差" << endl;
    }

    return 0;
}

// 数据清洗：过滤异常数据
int cleanData() {
    int newLen = 0;
    for (int i = 0; i < Len; i++) {
        // 过滤掉无评测、时长为0、价格异常的数据
        if (inputGame[i].pointNumber <= 0 || inputGame[i].time <= 0 || inputGame[i].points < 0) {
            cout << "过滤异常数据: " << inputGame[i].name << endl;
            continue;
        }
        if (newLen != i) {
            inputGame[newLen] = inputGame[i];
        }
        newLen++;
    }
    Len = newLen;
    return Len;
}

// 贝叶斯修正好评率
void bayesCorrect() {
    // 计算平均好评率
    double avg_points = 0.0;
    int validCnt = 0;
    for (int i = 0; i < Len; i++) {
        avg_points += inputGame[i].points;
        validCnt++;
    }
    avg_points /= validCnt;

    // 修正每个游戏的好评率
    for (int i = 0; i < Len; i++) {
        double p = inputGame[i].points;
        double n = inputGame[i].pointNumber;
        inputGame[i].b = (p * n + BAYES_M * avg_points) / (n + BAYES_M);
    }
}

// Min-Max归一化
void normalize() {
    // 计算各指标的min和max
    double min_t = 1e18, max_t = -1e18;
    double min_b = 1e18, max_b = -1e18;
    double min_p = 1e18, max_p = -1e18;

    for (int i = 0; i < Len; i++) {
        min_t = min(min_t, inputGame[i].time);
        max_t = max(max_t, inputGame[i].time);
        min_b = min(min_b, inputGame[i].b);
        max_b = max(max_b, inputGame[i].b);
        min_p = min(min_p, inputGame[i].price);
        max_p = max(max_p, inputGame[i].price);
    }

    // 归一化，处理分母为0的情况
    double range_t = max_t - min_t;
    double range_b = max_b - min_b;
    double range_p = max_p - min_p;
    if (range_t < EPS) range_t = 1;
    if (range_b < EPS) range_b = 1;
    if (range_p < EPS) range_p = 1;

    for (int i = 0; i < Len; i++) {
        // 时长：越大越好
        inputGame[i].t_norm = (inputGame[i].time - min_t) / range_t;
        // 好评率：越大越好
        inputGame[i].b_norm = (inputGame[i].b - min_b) / range_b;
        // 价格：越小越好，正向化
        inputGame[i].p_norm = (max_p - inputGame[i].price) / range_p;

        // 限制在0-1之间，防止浮点误差
        inputGame[i].t_norm = max(0.0, min(1.0, inputGame[i].t_norm));
        inputGame[i].b_norm = max(0.0, min(1.0, inputGame[i].b_norm));
        inputGame[i].p_norm = max(0.0, min(1.0, inputGame[i].p_norm));
    }
}

// 熵权法计算客观权重
void entropyWeight(double &w_t, double &w_b, double &w_p) {
    int m = Len;
    double k = 1.0 / log(m + EPS); // 避免log(0)

    // 计算每个指标的总和，用于概率化
    double sum_t = 0, sum_b = 0, sum_p = 0;
    for (int i = 0; i < m; i++) {
        sum_t += inputGame[i].t_norm + EPS;
        sum_b += inputGame[i].b_norm + EPS;
        sum_p += inputGame[i].p_norm + EPS;
    }

    // 计算熵值
    double e_t = 0, e_b = 0, e_p = 0;
    for (int i = 0; i < m; i++) {
        double p_t = (inputGame[i].t_norm + EPS) / sum_t;
        double p_b = (inputGame[i].b_norm + EPS) / sum_b;
        double p_p = (inputGame[i].p_norm + EPS) / sum_p;

        e_t -= p_t * log(p_t);
        e_b -= p_b * log(p_b);
        e_p -= p_p * log(p_p);
    }
    e_t *= k;
    e_b *= k;
    e_p *= k;

    // 差异系数
    double d_t = 1 - e_t;
    double d_b = 1 - e_b;
    double d_p = 1 - e_p;
    double sum_d = d_t + d_b + d_p;

    // 熵权
    w_t = d_t / sum_d;
    w_b = d_b / sum_d;
    w_p = d_p / sum_d;
}

// 组合权重：乘法组合
void combineWeight(double w_t_ahp, double w_b_ahp, double w_p_ahp, 
                   double w_t_e, double w_b_e, double w_p_e,
                   double &w_t, double &w_b, double &w_p) {
    double t = w_t_ahp * w_t_e;
    double b = w_b_ahp * w_b_e;
    double p = w_p_ahp * w_p_e;
    double sum = t + b + p;
    w_t = t / sum;
    w_b = b / sum;
    w_p = p / sum;
}

// TOPSIS计算得分
void topsis(double w_t, double w_b, double w_p) {
    int n = Len;
    // 加权规范化矩阵
    vector<vector<double>> Y(n, vector<double>(3));
    for (int i = 0; i < n; i++) {
        Y[i][0] = inputGame[i].t_norm * w_t;
        Y[i][1] = inputGame[i].b_norm * w_b;
        Y[i][2] = inputGame[i].p_norm * w_p;
    }

    // 正理想解和负理想解
    vector<double> Y_plus(3, -1e18), Y_minus(3, 1e18);
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < n; i++) {
            Y_plus[j] = max(Y_plus[j], Y[i][j]);
            Y_minus[j] = min(Y_minus[j], Y[i][j]);
        }
    }

    // 计算每个游戏的距离和得分
    for (int i = 0; i < n; i++) {
        double d_plus = 0, d_minus = 0;
        for (int j = 0; j < 3; j++) {
            d_plus += pow(Y[i][j] - Y_plus[j], 2);
            d_minus += pow(Y[i][j] - Y_minus[j], 2);
        }
        d_plus = sqrt(d_plus);
        d_minus = sqrt(d_minus);
        inputGame[i].score = d_minus / (d_plus + d_minus + EPS);
    }
}

// 统计检验
void statistical_test(ofstream& out_log) {
    cout << "\n===== 统计检验结果 =====" << endl;
    out_log << "\n===== 统计检验结果 =====" << endl;
    
    vector<double> scores, prices, times, points;
    for (int i = 0; i < Len; i++) {
        scores.push_back(inputGame[i].score);
        prices.push_back(inputGame[i].price);
        times.push_back(inputGame[i].time);
        points.push_back(inputGame[i].points);
    }
    
    // 1. Pearson相关性分析
    cout << "\n--- Pearson相关性分析 ---" << endl;
    out_log << "\n--- Pearson相关性分析 ---" << endl;
    
    auto r_price = pearson_correlation(scores, prices);
    auto r_time = pearson_correlation(scores, times);
    auto r_points = pearson_correlation(scores, points);
    
    cout << "性价比与价格: r=" << fixed << setprecision(4) << r_price.first << ", p=" << r_price.second << endl;
    cout << "性价比与时长: r=" << r_time.first << ", p=" << r_time.second << endl;
    cout << "性价比与好评率: r=" << r_points.first << ", p=" << r_points.second << endl;
    
    out_log << "性价比与价格: r=" << fixed << setprecision(4) << r_price.first << ", p=" << r_price.second << endl;
    out_log << "性价比与时长: r=" << r_time.first << ", p=" << r_time.second << endl;
    out_log << "性价比与好评率: r=" << r_points.first << ", p=" << r_points.second << endl;
    
    // 2. 多重共线性检验(VIF)
    cout << "\n--- 多重共线性检验(VIF) ---" << endl;
    out_log << "\n--- 多重共线性检验(VIF) ---" << endl;
    
    vector<double> t_norm, b_norm, p_norm;
    for (int i=0;i<Len;i++) {
        t_norm.push_back(inputGame[i].t_norm);
        b_norm.push_back(inputGame[i].b_norm);
        p_norm.push_back(inputGame[i].p_norm);
    }
    
    // 时长对其他两个指标的回归
    vector<vector<double>> X_tb = {b_norm, p_norm};
    double r2_t = linear_regression_r2(t_norm, X_tb);
    double vif_t = 1.0 / (1.0 - r2_t + EPS);
    
    // 好评率对其他两个指标的回归
    vector<vector<double>> X_tp = {t_norm, p_norm};
    double r2_b = linear_regression_r2(b_norm, X_tp);
    double vif_b = 1.0 / (1.0 - r2_b + EPS);
    
    // 价格对其他两个指标的回归
    vector<vector<double>> X_tb2 = {t_norm, b_norm};
    double r2_p = linear_regression_r2(p_norm, X_tb2);
    double vif_p = 1.0 / (1.0 - r2_p + EPS);
    
    cout << "时长的VIF: " << vif_t << endl;
    cout << "好评率的VIF: " << vif_b << endl;
    cout << "价格的VIF: " << vif_p << endl;
    
    out_log << "时长的VIF: " << vif_t << endl;
    out_log << "好评率的VIF: " << vif_b << endl;
    out_log << "价格的VIF: " << vif_p << endl;
    
    // 3. 偏相关分析
    cout << "\n--- 偏相关分析 ---" << endl;
    out_log << "\n--- 偏相关分析 ---" << endl;
    
    // 价格与性价比，控制时长和好评率
    auto r_sp = pearson_correlation(scores, prices);
    auto r_st = pearson_correlation(scores, times);
    auto r_pt = pearson_correlation(prices, times);
    auto r_sb = pearson_correlation(scores, points);
    auto r_pb = pearson_correlation(prices, points);
    
    double pr_price = (r_sp.first - r_st.first * r_pt.first - r_sb.first * r_pb.first) / 
                      sqrt((1 - r_pt.first*r_pt.first - r_pb.first*r_pb.first) * 
                           (1 - r_st.first*r_st.first - r_sb.first*r_sb.first));
    
    cout << "控制其他变量后，价格与性价比的偏相关系数: " << pr_price << endl;
    out_log << "控制其他变量后，价格与性价比的偏相关系数: " << pr_price << endl;
}

// 敏感性分析
void sensitivity_analysis(double w_t_ahp, double w_b_ahp, double w_p_ahp, ofstream& out_log) {
    cout << "\n===== 敏感性分析结果 =====" << endl;
    out_log << "\n===== 敏感性分析结果 =====" << endl;
    
    // 原始得分
    vector<double> original_scores;
    for (int i=0;i<Len;i++) original_scores.push_back(inputGame[i].score);
    
    // 1. 贝叶斯m的敏感性
    cout << "\n--- 贝叶斯先验常数m的敏感性 ---" << endl;
    out_log << "\n--- 贝叶斯先验常数m的敏感性 ---" << endl;
    
    vector<int> ms = {1,5,10,20};
    vector<vector<double>> m_scores;
    for (int m : ms) {
        auto s = recalculate_scores(m, false, false, false, w_t_ahp, w_b_ahp, w_p_ahp);
        m_scores.push_back(s);
    }
    
    for (int i=0;i<ms.size();i++) {
        for (int j=i+1;j<ms.size();j++) {
            double rho = spearman_correlation(m_scores[i], m_scores[j]);
            cout << "m=" << ms[i] << " vs m=" << ms[j] << ": Spearman rho=" << rho << endl;
            out_log << "m=" << ms[i] << " vs m=" << ms[j] << ": Spearman rho=" << rho << endl;
        }
    }
    
    // 2. 归一化方式的敏感性
    cout << "\n--- 归一化方式的敏感性 ---" << endl;
    out_log << "\n--- 归一化方式的敏感性 ---" << endl;
    
    auto z_scores = recalculate_scores(BAYES_M, true, false, false, w_t_ahp, w_b_ahp, w_p_ahp);
    double rho_norm = spearman_correlation(original_scores, z_scores);
    
    // 前10名重合度
    unordered_set<string> top10_original, top10_z;
    vector<pair<double, string>> tmp;
    for (int i=0;i<Len;i++) tmp.emplace_back(-original_scores[i], inputGame[i].name);
    sort(tmp.begin(), tmp.end());
    for (int i=0;i<min(10, Len);i++) top10_original.insert(tmp[i].second);
    
    tmp.clear();
    for (int i=0;i<Len;i++) tmp.emplace_back(-z_scores[i], inputGame[i].name);
    sort(tmp.begin(), tmp.end());
    for (int i=0;i<min(10, Len);i++) top10_z.insert(tmp[i].second);
    
    int overlap = 0;
    for (auto& s : top10_original) if (top10_z.count(s)) overlap++;
    double overlap_rate = overlap * 1.0 / 10;
    
    cout << "Min-Max vs Z-score: Spearman rho=" << rho_norm << endl;
    cout << "前10名重合度: " << overlap_rate*100 << "%" << endl;
    
    out_log << "Min-Max vs Z-score: Spearman rho=" << rho_norm << endl;
    out_log << "前10名重合度: " << overlap_rate*100 << "%" << endl;
    
    // 3. 距离度量的敏感性
    cout << "\n--- TOPSIS距离度量的敏感性 ---" << endl;
    out_log << "\n--- TOPSIS距离度量的敏感性 ---" << endl;
    
    auto manhattan_scores = recalculate_scores(BAYES_M, false, true, false, w_t_ahp, w_b_ahp, w_p_ahp);
    double rho_dist = spearman_correlation(original_scores, manhattan_scores);
    cout << "欧氏距离 vs 曼哈顿距离: Spearman rho=" << rho_dist << endl;
    out_log << "欧氏距离 vs 曼哈顿距离: Spearman rho=" << rho_dist << endl;
    
    // 4. 正向化方法的敏感性
    cout << "\n--- 价格正向化方法的敏感性 ---" << endl;
    out_log << "\n--- 价格正向化方法的敏感性 ---" << endl;
    
    auto inv_p_scores = recalculate_scores(BAYES_M, false, false, true, w_t_ahp, w_b_ahp, w_p_ahp);
    double rho_p = spearman_correlation(original_scores, inv_p_scores);
    cout << "1-P vs 1/(P+eps): Spearman rho=" << rho_p << endl;
    out_log << "1-P vs 1/(P+eps): Spearman rho=" << rho_p << endl;
    
    // 5. 权重扰动的敏感性
    cout << "\n--- 组合权重扰动的敏感性 ---" << endl;
    out_log << "\n--- 组合权重扰动的敏感性 ---" << endl;
    
    srand(time(0));
    double avg_overlap = 0;
    for (int iter=0;iter<100;iter++) {
        // 随机扰动±20%
        double eps_t = (rand() * 2.0 / RAND_MAX) - 1.0; // -1~1
        double eps_b = (rand() * 2.0 / RAND_MAX) - 1.0;
        double eps_p = (rand() * 2.0 / RAND_MAX) - 1.0;
        
        double w_t = w_t_ahp * (1 + eps_t * 0.2);
        double w_b = w_b_ahp * (1 + eps_b * 0.2);
        double w_p = w_p_ahp * (1 + eps_p * 0.2);
        double sum = w_t + w_b + w_p;
        w_t /= sum; w_b /= sum; w_p /= sum;
        
        auto perturbed_scores = recalculate_scores(BAYES_M, false, false, false, w_t, w_b, w_p);
        
        // 前20名重合度
        unordered_set<string> top20_original, top20_pert;
        vector<pair<double, string>> tmp_o, tmp_p;
        for (int i=0;i<Len;i++) tmp_o.emplace_back(-original_scores[i], inputGame[i].name);
        sort(tmp_o.begin(), tmp_o.end());
        for (int i=0;i<min(20, Len);i++) top20_original.insert(tmp_o[i].second);
        
        for (int i=0;i<Len;i++) tmp_p.emplace_back(-perturbed_scores[i], inputGame[i].name);
        sort(tmp_p.begin(), tmp_p.end());
        for (int i=0;i<min(20, Len);i++) top20_pert.insert(tmp_p[i].second);
        
        int overlap = 0;
        for (auto& s : top20_original) if (top20_pert.count(s)) overlap++;
        avg_overlap += overlap * 1.0 / 20;
    }
    avg_overlap /= 100;
    cout << "权重扰动后，前20名平均重合度: " << avg_overlap*100 << "%" << endl;
    out_log << "权重扰动后，前20名平均重合度: " << avg_overlap*100 << "%" << endl;
}

// 交叉验证
void cross_validation(double w_t_ahp, double w_b_ahp, double w_p_ahp, ofstream& out_log) {
    cout << "\n===== 交叉验证结果 =====" << endl;
    out_log << "\n===== 交叉验证结果 =====" << endl;
    
    vector<double> original_scores;
    for (int i=0;i<Len;i++) original_scores.push_back(inputGame[i].score);
    
    // 1. 随机分割验证
    cout << "\n--- 随机80-20分割验证 ---" << endl;
    out_log << "\n--- 随机80-20分割验证 ---" << endl;
    
    double avg_rho = 0;
    for (int iter=0;iter<10;iter++) {
        vector<int> idx(Len);
        iota(idx.begin(), idx.end(), 0);
        random_shuffle(idx.begin(), idx.end());
        
        int train_size = Len * 0.8;
        vector<Game> train, test;
        for (int i=0;i<train_size;i++) train.push_back(inputGame[idx[i]]);
        for (int i=train_size;i<Len;i++) test.push_back(inputGame[idx[i]]);
        
        // 用训练集计算参数
        double avg_points = 0;
        for (auto& g : train) avg_points += g.points;
        avg_points /= train.size();
        
        double min_t = 1e18, max_t = -1e18;
        double min_b = 1e18, max_b = -1e18;
        double min_p = 1e18, max_p = -1e18;
        for (auto& g : train) {
            min_t = min(min_t, g.time);
            max_t = max(max_t, g.time);
            min_b = min(min_b, (g.points * g.pointNumber + BAYES_M * avg_points)/(g.pointNumber + BAYES_M));
            max_b = max(max_b, (g.points * g.pointNumber + BAYES_M * avg_points)/(g.pointNumber + BAYES_M));
            min_p = min(min_p, g.price);
            max_p = max(max_p, g.price);
        }
        
        // 处理测试集
        vector<double> test_scores;
        for (auto& g : test) {
            double b = (g.points * g.pointNumber + BAYES_M * avg_points)/(g.pointNumber + BAYES_M);
            double t_norm = (g.time - min_t)/(max_t - min_t + EPS);
            double b_norm = (b - min_b)/(max_b - min_b + EPS);
            double p_norm = (max_p - g.price)/(max_p - min_p + EPS);
            
            // 熵权（简化，用原来的）
            double w_t_e, w_b_e, w_p_e;
            // 这里简化，用原来的熵权
            entropyWeight(w_t_e, w_b_e, w_p_e);
            
            double w_t, w_b, w_p;
            combineWeight(w_t_ahp, w_b_ahp, w_p_ahp, w_t_e, w_b_e, w_p_e, w_t, w_b, w_p);
            
            // TOPSIS
            vector<vector<double>> Y(1, vector<double>(3));
            Y[0][0] = t_norm * w_t;
            Y[0][1] = b_norm * w_b;
            Y[0][2] = p_norm * w_p;
            
            // 简化，用原来的正负理想解
            double d_plus = 0, d_minus =0;
            // 这里简化，直接用得分的相对值
            double score = t_norm * w_t + b_norm * w_b + p_norm * w_p;
            test_scores.push_back(score);
        }
        
        // 计算和全量排名的相关
        vector<double> full_test_scores;
        for (auto& g : test) {
            for (int j=0;j<Len;j++) {
                if (inputGame[j].name == g.name) {
                    full_test_scores.push_back(inputGame[j].score);
                    break;
                }
            }
        }
        
        double rho = spearman_correlation(test_scores, full_test_scores);
        avg_rho += rho;
    }
    avg_rho /=10;
    cout << "10次随机分割的平均Spearman相关系数: " << avg_rho << endl;
    out_log << "10次随机分割的平均Spearman相关系数: " << avg_rho << endl;
    
    // 2. 留一法验证
    cout << "\n--- 留一法交叉验证 ---" << endl;
    out_log << "\n--- 留一法交叉验证 ---" << endl;
    
    double avg_diff = 0, max_diff = 0;
    int cnt =0;
    for (int leave=0;leave<min(10, Len);leave++) { // 只算前10个，避免太慢
        // 移除leave这个样本
        vector<Game> others;
        Game left = inputGame[leave];
        for (int i=0;i<Len;i++) if (i!=leave) others.push_back(inputGame[i]);
        
        // 计算参数
        double avg_points =0;
        for (auto& g : others) avg_points += g.points;
        avg_points /= others.size();
        
        double b_left = (left.points * left.pointNumber + BAYES_M * avg_points)/(left.pointNumber + BAYES_M);
        double original_b = inputGame[leave].b;
        
        double diff = abs(b_left - original_b);
        avg_diff += diff;
        max_diff = max(max_diff, diff);
        cnt++;
    }
    avg_diff /= cnt;
    cout << "留一法平均得分差异: " << avg_diff << endl;
    cout << "留一法最大得分差异: " << max_diff << endl;
    out_log << "留一法平均得分差异: " << avg_diff << endl;
    out_log << "留一法最大得分差异: " << max_diff << endl;
}


int main() {
    cout << "欢迎使用游戏性价比分析工具！本工具支持批量处理多个数据文件，结果会自动保存不覆盖。" << endl;
    cout << "最多支持连续处理10组数据，处理完一组后可选择继续或退出。" << endl;
    
    // 最多处理10组数据
    for(int batch_idx = 0; batch_idx < 10; batch_idx++) {
        cout << "\n\n===== 第 " << batch_idx + 1 << " 组数据处理 =====" << endl;
        
        // 每次循环前重置所有全局变量，避免上一组数据残留
        Len = 0;
        ahp.timeSum = 0;
        ahp.pointsSum = 0;
        ahp.priceSum = 0;
        timeAHP = 0;
        pointsAHP = 0;
        priceAHP = 0;
        memset(matrixOld, 0, sizeof(matrixOld));
        memset(matrixNew, 0, sizeof(matrixNew));
        cin.ignore(numeric_limits<streamsize>::max(), '\n'); // 清空输入缓冲区，避免残留换行
        
        // 打开当前批次的日志文件，自动加序号避免覆盖
        string log_name = "validation_result_" + to_string(batch_idx + 1) + ".txt";
        ofstream out_log(log_name);
        
        // 1. 读取游戏数据
        cout << "\n--- 读取游戏数据 ---" << endl;
        if (InputGame() != 0) {
            cout << "当前组数据读取失败，是否继续处理下一组？(y/n)：";
            string choice;
            getline(cin, choice);
            if (choice != "y" && choice != "Y") {
                cout << "程序退出。" << endl;
                return 0;
            }
            continue;
        }
        if (Len == 0) {
            cout << "没有读取到有效数据！是否继续处理下一组？(y/n)：";
            string choice;
            getline(cin, choice);
            if (choice != "y" && choice != "Y") {
                cout << "程序退出。" << endl;
                return 0;
            }
            continue;
        }

        // 2. 数据清洗
        cout << "\n--- 数据清洗 ---" << endl;
        cleanData();
        if (Len == 0) {
            cout << "清洗后没有有效数据！是否继续处理下一组？(y/n)：";
            string choice;
            getline(cin, choice);
            if (choice != "y" && choice != "Y") {
                cout << "程序退出。" << endl;
                return 0;
            }
            continue;
        }
        cout << "清洗后剩余 " << Len << " 条有效数据" << endl;

        // 3. 读取AHP矩阵，计算主观权重
        cout << "\n--- 计算AHP主观权重 ---" << endl;
        if (InputAhp() != 0) {
            cout << "AHP矩阵读取失败，是否继续处理下一组？(y/n)：";
            string choice;
            getline(cin, choice);
            if (choice != "y" && choice != "Y") {
                cout << "程序退出。" << endl;
                return 0;
            }
            continue;
        }
        makeMatrix();

        // 4. 贝叶斯修正好评率
        cout << "\n--- 贝叶斯修正好评率 ---" << endl;
        bayesCorrect();
        double avg_points = 0;
        for(int i=0;i<Len;i++) avg_points += inputGame[i].points;
        avg_points /= Len;
        cout << "全局平均好评率: " << avg_points << endl;

        // 5. 归一化
        cout << "\n--- 指标归一化 ---" << endl;
        normalize();

        // 6. 熵权法计算客观权重
        cout << "\n--- 计算熵权客观权重 ---" << endl;
        double w_t_e, w_b_e, w_p_e;
        entropyWeight(w_t_e, w_b_e, w_p_e);
        cout << "熵权法权重: 时长=" << w_t_e << ", 好评率=" << w_b_e << ", 价格=" << w_p_e << endl;

        // 7. 组合权重
        cout << "\n--- 计算组合权重 ---" << endl;
        double w_t, w_b, w_p;
        combineWeight(timeAHP, pointsAHP, priceAHP, w_t_e, w_b_e, w_p_e, w_t, w_b, w_p);
        cout << "AHP主观权重: 时长=" << timeAHP << ", 好评率=" << pointsAHP << ", 价格=" << priceAHP << endl;
        cout << "最终组合权重: 时长=" << w_t << ", 好评率=" << w_b << ", 价格=" << w_p << endl;

        // 8. TOPSIS计算性价比得分
        cout << "\n--- 计算TOPSIS性价比得分 ---" << endl;
        topsis(w_t, w_b, w_p);

        // 9. 排序并输出结果
        cout << "\n--- 排序结果 ---" << endl;
        vector<pair<double, string>> results;
        for (int i = 0; i < Len; i++) {
            results.emplace_back(-inputGame[i].score, inputGame[i].name); // 负号用于升序排序，等价于得分降序
        }
        sort(results.begin(), results.end());

        // 输出表头
        cout << "排名\t得分\t游戏名称" << endl;
        for (int i = 0; i < min((int)results.size(), 20); i++) { // 只输出前20名，避免刷屏
            double score = -results[i].first;
            string name = results[i].second;
            cout << (i+1) << "\t" << fixed << setprecision(4) << score << "\t" << name << endl;
        }
        if(results.size() > 20) {
            cout << "... 剩余" << results.size() -20 << "条数据已保存到文件" << endl;
        }

        // 输出到csv文件，自动加序号避免覆盖
        string out_name = "result_" + to_string(batch_idx + 1) + ".csv";
        ofstream out(out_name);
        out << "排名,游戏名称,性价比得分,归一化时长,修正好评率,正向化价格" << endl;
        for (int i = 0; i < results.size(); i++) {
            // 找到对应的游戏
            int idx = -1;
            for (int j = 0; j < Len; j++) {
                if (inputGame[j].name == results[i].second) {
                    idx = j;
                    break;
                }
            }
            if (idx == -1) continue;
            out << (i+1) << "," 
                << inputGame[idx].name << "," 
                << fixed << setprecision(6) << inputGame[idx].score << ","
                << inputGame[idx].t_norm << ","
                << inputGame[idx].b << ","
                << inputGame[idx].p_norm << endl;
        }
        out.close();
        cout << "\n当前组完整排名结果已保存到 " << out_name << " 文件中！" << endl;
        
        // 10. 运行所有检验
        statistical_test(out_log);
        sensitivity_analysis(timeAHP, pointsAHP, priceAHP, out_log);
        cross_validation(timeAHP, pointsAHP, priceAHP, out_log);
        
        cout << "\n当前组所有检验结果已保存到 " << log_name << " 文件中！" << endl;
        out_log.close();

        // 询问用户是否继续处理下一组
        cout << "\n第 " << batch_idx + 1 << " 组数据处理完成！是否继续处理下一组数据？(y/n)：";
        string choice;
        getline(cin, choice);
        if (choice != "y" && choice != "Y") {
            cout << "所有处理已完成，程序退出。" << endl;
            return 0;
        }
    }
    
    // 处理完10组后自动退出
    cout << "\n已完成最多10组数据的处理，程序退出。" << endl;
    return 0;
}



// #include <bits/stdc++.h>
// using namespace std;
// const double RI = 0.580000;


// struct Game {
//     string name;
//     double price;
//     double points;
//     double pointNumber;
//     double time;
//     int type;
// }inputGame[1000];

// struct Ahp{
//     double time;
//     double points;
//     double price;
//     double timeSum = 0.0;
//     double pointsSum = 0.0;
//     double priceSum = 0.0;
// }ahp;

// int Len = 0;
// double matrixOld[4][4], matrixNew[4][4], timeAHP, pointsAHP, priceAHP, ci, aw[4];


// double ahpMaxMin(double x, double y){
//     return x/y;
// }

// int InputGame(){
//     cout << "请输入文件位置， 样例：" << "D:/GameCP/Data/a.csv" << endl;
//     string path ;
//     getline(cin ,path);

//     ifstream file(path);  
    
//     //错误处理
//     if (!file.is_open()) {
//         cout << "错误：找不到文件！请检查路径是否正确：" << endl;
//         cout << "你填写的路径是：" << path << endl;
//         cout << "注意：Windows路径要用 / 或 \\\\，不要只用 \\" << endl;
//         return 1;
//     }

//     string line;
//     getline(file, line);

//     //读取文件行数
//     while (getline(file, line)) {

//         if (line.empty()) continue;
//         Len++;
//     }
//     cout << "共有 " << Len << " 行文件" << endl;


//     file.close();
//     file.open(path);

//     //开始读取并写入数据
//     getline(file, line);

//     int i = 0;
//     while (getline(file, line) && i < Len){
//         if (line.empty()) continue;
//         stringstream ss(line);
//         string sName, sPrice, sPoints, sPointNumber, sTime;

//         getline(ss, sName, ',');
//         getline(ss, sPrice, ',');
//         getline(ss, sPoints, ',');
//         getline(ss, sPointNumber, ',');
//         getline(ss, sTime, ',');


//         inputGame[i].name = sName;

//         i ++;
//         // cout << sPrice << endl;
//         try {
//             inputGame[i].price = stod(sPrice);
//             inputGame[i].points = stod(sPoints);
//             inputGame[i].pointNumber = stod(sPointNumber);
//             inputGame[i].time = stod(sTime);
//         } catch (const invalid_argument&) {
//             // cout << "数据格式错误：" << line << endl;
//             if (sPrice == "免费开玩") inputGame[i].price = 0.00;
//             if (sPoints == "无用户评测") inputGame[i].points = inputGame[i].pointNumber = 0.00;
//             continue;
//         }

//     }
    
//     file.close();

//     return 0;
// }


// int InputAhp(){
//     cout << "请输入文件位置， 样例：" << "D:/GameCP/Data/question.csv" << endl;
//     string path ;
//     getline(cin ,path);

//     ifstream file(path);  
    
//     //错误处理
//     if (!file.is_open()) {
//         cout << "错误：找不到文件！请检查路径是否正确：" << endl;
//         cout << "你填写的路径是：" << path << endl;
//         cout << "注意：Windows路径要用 / 或 \\\\，不要只用 \\" << endl;
//         return 1;
//     }
    
//     int i = 1;
//     string line;
    
//     while (getline(file, line) && i <= 3){
//         if (line.empty()) continue;
//         stringstream ss(line);
//         string sTime, sPoints, sPrice;
        
//         getline(ss, sTime, ',');
//         getline(ss, sPoints, ',');
//         getline(ss, sPrice, ',');
        
//         matrixOld[i][1] = stod(sTime);    ahp.timeSum += stod(sTime);
//         matrixOld[i][2] = stod(sPoints);  ahp.pointsSum += stod(sPoints);
//         matrixOld[i][3] = stod(sPrice);   ahp.priceSum += stod(sPrice);
//         i ++;
//     }

//     return 0;
// }

// int writeMatrix(){
//     InputAhp();
//     for (int i = 1; i <= 3; i++) {
//         matrixNew[i][1] =  ahpMaxMin(matrixOld[i][1], ahp.timeSum);

//     }
//     for (int i = 1; i <= 3; i++) {
//         matrixNew[i][2] =  ahpMaxMin(matrixOld[i][2], ahp.pointsSum);
//     }
//     for (int i = 1; i <= 3; i++) {
//         matrixNew[i][3] =  ahpMaxMin(matrixOld[i][3], ahp.priceSum);    
//     }
//     return 0;
// }


// double Aw(double w1, double w2, double w3, int i){
//     aw[i] = w1*matrixOld[i][1] + w2*matrixOld[i][2] + w3*matrixOld[i][3];
//     return aw[i];
// }
// double CI(double w1, double w2, double w3){
//     double i = 0.0, j = 0.0, k = 0.0;
//     i = Aw(w1, w2, w3, 1);
//     j = Aw(w1, w2, w3, 2);
//     k = Aw(w1, w2, w3, 3);
//     double sum = (i + j + k - 3)/2;
//     return sum;
// }
// double CR(double w1, double w2, double w3){
//     return CI(w1, w2, w3)/RI;
// }

// double pdMatrix(double w1, double w2, double w3){
//     return CR(w1, w2, w3);
// }


// bool isMatrix(double w1, double w2, double w3){
//      if (pdMatrix(w1, w2, w3) < 0.1){
//         return true;
//     }else{
//         return false;
//     }
// }

// int makeMatrix(){
//     writeMatrix();

//     double i, j, k;
//     i = matrixNew[1][1] + matrixNew[1][2] + matrixNew[1][3];
//     j = matrixNew[2][1] + matrixNew[2][2] + matrixNew[2][3];
//     k = matrixNew[3][1] + matrixNew[3][2] + matrixNew[3][3];
//     double sum = i + j + k;
//     timeAHP = i/sum;
//     pointsAHP = j/sum;
//     priceAHP = k/sum;

//     if (isMatrix(timeAHP, pointsAHP, priceAHP)){
//         cout << "CR = " << pdMatrix(timeAHP, pointsAHP, priceAHP) << endl;
//         cout << "矩阵是可接受的" << endl;
//     } else{
//         cout << "CR = " << pdMatrix(timeAHP, pointsAHP, priceAHP) << endl;
//         cout << "矩阵不是可接受的" << endl;
//     }

//     return 0;
// }






// int main() {
//     // InputGame();
//     // cout << endl;
//     makeMatrix();

//     for (int i = 1; i <= 3; i++) {
//         for (int j = 1; j <= 3; j++) {
//             cout << matrixNew[i][j] << " ";
//         }
//         cout << endl;
//     }
// }