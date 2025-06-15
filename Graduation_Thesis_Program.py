import numpy as np
import cupy as cp
import itertools
import time

cp.set_printoptions(threshold=cp.inf)

# setting--------------------------------
delta = 0.1  # 0.1
t2 = 0.03 #0.5- 4 相関の大きさ
snrdB_iteration=100
snrdB_default=-4
SNR_INTERVAL=1
word_error_iteration=50
m_range=5
order=0

n = 24  # 次元数
k = 12

G = cp.array([
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
                            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
                            ])
#setting_end--------------------------------

def main():




    with open('data/osd.txt', 'w') as f:
        print(f"初期設定-------------------------",file=f)
        print(f"Order:{order}",file=f)
        print(f"m_range:{m_range}",file=f)
        print(f"符号長n:{n}",file=f)
        print(f"情報記号数:{k}",file=f)
        print(f"delta:{delta}",file=f)
        print(f"t2:{t2}",file=f)
        print(f"生成行列G\n{G}",file=f)
        print(f"情報記号はランダム生成",file=f)
        print(f"snrdB_iteration:{snrdB_iteration}",file=f)
        print(f"snrdb_default:{snrdB_default}",file=f)
        print(f"word_error_iteration:{word_error_iteration}",file=f)
        print(f"初期設定end-----------------------",file=f)


    with open("data/osd_onlydata.txt", "w") as f2:
        print(f"初期設定-------------------------",file=f2)
        print(f"Order:{order}",file=f2)
        print(f"m_range:{m_range}",file=f2)
        print(f"符号長n:{n}",file=f2)
        print(f"情報記号数:{k}",file=f2)
        print(f"delta:{delta}",file=f2)
        print(f"t2:{t2}",file=f2)
        print(f"生成行列G\n{G}",file=f2)
        print(f"情報記号はランダム生成",file=f2)
        print(f"snrdB_iteration:{snrdB_iteration}",file=f2)
        print(f"snrdb_default:{snrdB_default}",file=f2)
        print(f"word_error_iteration:{word_error_iteration}",file=f2)
        print(f"初期設定end-----------------------",file=f2)

        pass

    #------------------------線形符号計算--------------------------
    # values=[0,1]
    # m_combinations = cp.array(list(itertools.product(values, repeat=k)))
    # candidate_codeword = codeword_create(G, m_combinations)
    # candidate_codeword_bpsks=cp.where(candidate_codeword==0,1,-1)
    #--------------------------------------------------------
    #-----------------------D(組み合わせ)の計算-------------------
    values=[-1,1]
    mxTwoCombinations=np.array(list(itertools.product(values,repeat=2*m_range)))
    Minus1Combinations = np.insert(mxTwoCombinations, m_range, -1, axis=1)
    Minus1Combinations=cp.array(Minus1Combinations)
    print(f"MinusCombinations:\n{Minus1Combinations}")
    #------------------------------------------------------------



    #------------------------Orderに従って、error位置を作る-------------------
    all_combinations=[]
    for i in range(order+1):
        arr = [0] * k  # 全て0で初期化
        arr[:i] = [1] * i  # 最初のk個を1にする
        combinations=generate_combinations(arr)
        print(f"combinations:\n{combinations}")
        all_combinations=all_combinations+combinations
    error_positions=cp.array(all_combinations)
    #---------------------------------------------------------

    

    for i in range(snrdB_iteration):
        snrdB=i*SNR_INTERVAL+snrdB_default
        t1 = create_t_1_from_snrdB(snrdB, n, k)

        #相関通信路
        sigma = create_sigma(n, t1)
        print(f"sigma:\n{sigma}")
        #相関通信路end

        #無相関通信路
        # sigma=t1*cp.eye(n) #単位行列
        #無相関通信路end


        word_error_count = 0
        # bit_error_count=0
        iteration_count = 0
        while word_error_count < word_error_iteration:
            # time_start1 = time.time()
            # G = generator_matrix_random(n, k)
            m = m_create(k)
            x = codeword_create(G, m)
            r = received_sequence_create(x, sigma)

            log_Posterior_Probability_RatioS=logPosteriorProbabilityRatioS(r,m_range,t1,Minus1Combinations)
            # print(f"log_Posterior__Probability_RatioS:\n{log_Posterior_Probability_RatioS}")

            pi1=sortPermutationMatrixByConfidence(log_Posterior_Probability_RatioS)
            

            y=hardDecisionFromLogPosteriorProbabilityRatioS(log_Posterior_Probability_RatioS)
            #--------------from received sequence----------------------------------
            # pi1=sortPermutationMatrixByReceivedSequence(r)
            # y=hardDecisionFromReceivedSequence(r)
            #---------------------------------------------------------------

            y2=cp.dot(y,pi1)%2
            # print(f"ratio:\n{cp.dot(log_Posterior_Probability_RatioS,pi1)}")
            # print(f"y2:\n{y2}")
            # print(f"x:\n{cp.dot(x,pi1)%2}")
            G2=cp.dot(G,pi1)%2

            G2_copy=G2.copy()

            U,pi2=gaussian_elimination(G2_copy)

            y3=cp.dot(y2,pi2)%2
            # print(f"y3:\n{y3}")
            # print(f"x:\n{cp.dot(cp.dot(x,pi1)%2,pi2)}")
            
            G3=cp.dot(cp.dot(U,G2)%2,pi2)

            u0=y3[0:k]

            x_estimate=OSD(u0,G3,r,sigma,pi1,pi2,error_positions,t1)

            

            # r=cp.array([0.07529169 ,0.8168622  ,1.34055932 ,1.22124756])
            # x=cp.array([1 ,1 ,1 ,0])
            # r=cp.array([-0.56528307, -0.30235741,  0.14088456, -1.86911216])
            # G=cp.array([[1, 0, 1, 0, 0],
            #             [1, 0, 0, 1, 1],
            #             [1, 1, 1, 1, 0]])

            # m=cp.array([1,1,0])
            # v_bpsk = codeword_create(G, cp.array([m]))
            # r=cp.array([46469.37048167, 62456.81929849, 73647.04221187, 76493.30585836,
            #             68700.04019895])
            

            # correct_loglikehood = logpdf(r, v_bpsk, sigma)
            print("----------------------------------")
            print(f"snrdB:{snrdB}")
            print(f"情報記号:{m}")
            print(f"送信符号語:{x}")

            # max_loglikehood, x_estimate=batch_mle_calculate(G,k,r,sigma,limited_memory)
            # max_loglikehood, x_estimate=pre_batch_max_loglikehood_estimate_calculate(G, k, r, sigma)
            # print(f"v_bpsk:\n{v_bpsk}")
            # print(f"estimate_v:\n{x_estimate}")

            #--------WordErrorRate-----------
            if cp.all(x == x_estimate):
                print(f"復号成功")
            else:
                print(f"復号失敗")
                word_error_count +=1
            #---------------------------------
            #---------BitErrorRate-----------
            # differentBit=cp.sum(x != x_estimate)
            # if differentBit==0:
            #     print(f"復号成功")
            # else:
            #     print(f"復号失敗")
            #     bit_error_count +=differentBit
            #---------------------------------


            # time_end1 = time.time()
            # print(f"出力尤度:{max_loglikehood}")
            print(f"推定符号語:{x_estimate}")
            print(f"現在までの復号誤り個数:{word_error_count}")
            # print(f"現在までの復号bit誤り個数:{bit_error_count}")
            print(f"現在までの反復回数:{iteration_count+1}")
            # print(f"時間:{time_end1-time_start1}")

            iteration_count +=1

        print(f"wer:{word_error_count/iteration_count}")
        # print(f"ber:{bit_error_count/(iteration_count*n)}")
        with open("data/osd.txt","a") as f:
            print(f"------------------------------",file=f)
            print(f"snrdB:{snrdB}",file=f)
            print(f"{iteration_count}回目,合計誤り回数:{word_error_count}",file=f)
            # print(f"{iteration_count}回目,合計bit誤り回数:{bit_error_count}",file=f)
            print(f"WER:{word_error_count/iteration_count}",file=f)
            # print(f"BER:{bit_error_count/(iteration_count*n)}",file=f)
            # print(f"Minute_Value_n:\n{Minute_Value_n}",file=f)
        with open("data/osd_onlydata.txt","a")as f2:
            print(f"{word_error_count/iteration_count},",file=f2)
            # print(f"{bit_error_count/(iteration_count*n)},",file=f2)

    # ------------------------------------------

    return 0

def generator_matrix_random(n, k):
    G = cp.random.randint(0, 2, size=(k,n))
    return G


def m_create(k):
    return cp.random.randint(0, 2, size=(k,))


def codeword_create(G, m):
    return cp.dot(m, G) % 2


def create_t_1_from_snrdB(snrdB, n, k):
    Eb = n/k  # =1/R
    N0 = Eb/(cp.power(10, snrdB/10))
    t1 = N0/2  # 分散はN0/2らしい
    print(f"Eb:{Eb}")
    print(f"N0:{N0}")
    print(f"t1(分散):{t1}")
    return t1

# def variance_generate(snrdB,n,k):
    R=k/n
    snr=10**(snrdB/10)
    variance=1/(2*R*snr)
    return variance

def noise_create(sigma):
    n = sigma.shape[0]
    mu = cp.zeros(n)
    noises = cp.random.multivariate_normal(mu, sigma, method='cholesky',dtype=np.float64,check_valid='raise',tol=1e-08)

    return noises


def received_sequence_create(x, sigma):
    x_bpsk=cp.where(x==0,1,-1)
    z = noise_create(sigma)
    # z=cp.array([1.2,0.8,1,0.7,0.5])
    # z=cp.array([-2.3,-1.9,-2.0,-2.4,-2.6])
    # z=np.array(z)
    r = x_bpsk+z
    # print(f"v:\n{v}")
    # print(f"noises:\n{z}")
    # print(f"r:\n{r}")
    return r


def k_ij(i, j, t1):  # カーネル
    # setting------------
    # t_1 = t1  # t_1=N_0なので、snr=1/N_0
    # delta = 0.1  # 0.1
    # t_2 = 0.5  # 0.5- 4 相関の大きさ
    # -------------------
    return t1*cp.exp((-cp.power(delta*i-delta*j, 2))/t2)


def create_sigma(n, t1):
    sigma = cp.empty((n, n))
    for i in range(n):
        for j in range(n):
            sigma[i, j] = k_ij(i, j, t1)
    return sigma

def multi_logpdf(x, means, cov):
    L = cp.linalg.cholesky(cov)
    k=means.shape[0]
    n=means.shape[1]
    dev = x-means 
    dev=dev.reshape(k,n,1) 
    L_expanded = cp.tile(L, (k, 1, 1))
    z = cp.linalg.solve(L_expanded, dev)
    maha=z.transpose(0, 2, 1)@z
    result=cp.exp(-0.5*maha)
    
    return result

# def MahalanobisDistances(ys,xs,cov):
    devs=ys-xs
    cov_inv=cp.linalg.pinv(cov)
    maha=cp.sum((devs@cov_inv)*devs,axis=1)

    return maha

def MahalanobisDistancebyCholesky(ys,xs,cov):
    # start=time.time()

    L = cp.linalg.cholesky(cov)
    k=xs.shape[0]
    n=xs.shape[1]
    devs = ys-xs  
    devs=devs.reshape(k,n,1) 
    L_expanded = cp.tile(L, (k, 1, 1))

    z = cp.linalg.solve(L_expanded, devs)
    maha=z.transpose(0, 2, 1)@z

    # end=time.time()

    # print(f"MahalanobisDistancebyCholesky_time:\n{end-start}")

    

    return maha
    
# def EuclideanDistance(y, x, variance):

    euclideans=(y-x)**2
    result=cp.exp((-1*euclideans)/(2*variance))
    return cp.sum(result,axis=1)
    

# def OnlyEuclideanDistance(y,x,variance):
    euclideans=(y-x)**2
    # result=(-1*euclideans)/(2*variance)
    # result=cp.exp((-1*euclideans)/(2*variance))
    # return cp.sum(euclideans,axis=1)
    # return cp.sum(result,axis=1)
    return euclideans




def LogPosteriorOdds(sliced_r,t1,i,m,Minus1Combinations):

    #sigmaに加える微小値は考えて
    #有色通信路-----------------------------------------------------------------------
    sigma=create_sigma(sliced_r.shape[0],t1)

    #----------------------------------------------------------------------------

    #白色通信路------------------------
    # sigma=t1*cp.eye(sliced_r.shape[0])
    # print(f"sigma:\n{sigma}")
    #-----------------------------------

    # print(f"Minus1Combinations:\n{Minus1Combinations}")

    # print(f"candidate_codeword_bpsk:\n{candidate_codeword_bpsk}")
    if i-m<=0:
        Plus1Combinations=Minus1Combinations.copy()
        Plus1Combinations[:,i]=1
    else:
        Plus1Combinations=Minus1Combinations.copy()
        Plus1Combinations[:,m]=1

    # print(f"ifPlus1Combinations:\n{Plus1Combinations}")


    # print(f"combinations_plus1:\n{combinations_plus1}")
    # print(f"combinations_minus1:\n{combinations_minus1}")

    
    
        
    #-----------------------------------------------------
    PosteriorOdds_plus1=multi_logpdf(sliced_r,Plus1Combinations,sigma)
    PosteriorOdds_minus1=multi_logpdf(sliced_r,Minus1Combinations,sigma)

    LogPosteriorOdds=cp.log(cp.sum(PosteriorOdds_plus1)) -cp.log(cp.sum(PosteriorOdds_minus1))
    # print(f"sum_PoteriorOdds_plus1:\n{cp.sum(PosteriorOdds_plus1)}")
    # print(f"sum_PoteriorOdds_minus1:\n{cp.sum(PosteriorOdds_minus1)}")
    


    


    # print(f"sliced_r,{sliced_r}")
    # print(f"sigma:\n{sigma}")
    # print(f"combination_plus1:\n{combinations_plus1}")
    # print(f"combination_minus1:\n{combinations_minus1}")

    # print(f"sum:\n{cp.sum(PosteriorOdds_plus1)}")
    # print(f"PosteriorOdds_plus1:\n{PosteriorOdds_plus1}")
    # print(f"PosteriorOdds_minus1:\n{PosteriorOdds_minus1}")
    
    # print(f"cp.sum(PosteriorOdds_plus1):\n{cp.sum(PosteriorOdds_plus1)}")

    # print(f"combinations:\n{combinations}")
    # print(f"combination_plus1:\n{combinations_plus1}")
    # print(f"combination_minus1:\n{combinations_minus1}")

    # print(f"LogPosteriorOdds:\n{LogPosteriorOdds}")

    return LogPosteriorOdds

def logPosteriorProbabilityRatioS(r,m,t1,Minus1Combinations):
    LogPosteriorProbabilitieS=cp.empty(len(r))
    
    total_time=0
    for i in range(len(r)):
    
        start = max(i - m, 0)  # インデックス範囲を 0 以上に制限
        end = min(i + m + 1, len(r))  # インデックス範囲を len(r) 以下に制限
        start2=min(i,m)
        end2=min(len(r)-i-1,m)
        sliced_r = r[start:end]
        way_start=time.time()    
        log_posterior_odds=LogPosteriorOdds(sliced_r,t1,i,m,Minus1Combinations[:,m-start2:m+end2+1])
        LogPosteriorProbabilitieS[i]=log_posterior_odds
        way_end=time.time()
        # if end-start==2*m+1:
        #     log_posterior_odds=LogPosteriorOdds(sliced_r,t1,i,m,m_combinations)
        # else:
        #     except_m_combinations=cp.array(list(itertools.product(values, repeat=end-start-1)))
        #     log_posterior_odds=LogPosteriorOdds(sliced_r,t1,i,m,except_m_combinations)

        # if log_posterior_odds>0:
        #     estimated_codeword[i]=0
        # elif log_posterior_odds<0:
        #     estimated_codeword[i]=1
        # else:
        #     raise ValueError("エラーが発生しました:対数事後確率比が０になりました。")

    
    # print(f"estimated_codeword:\n{estimated_codeword}")
    # return 0
    # way_end=time.time()
    # total_time=total_time+(way_end-way_start)
    # print(f"time:\n{total_time}")
    return LogPosteriorProbabilitieS

def sortPermutationMatrixByConfidence(r):
    n=r.shape[0]
    identity_matrix=cp.eye(n)
    sorted_indices=cp.argsort(-cp.abs(r))
    permutation_matrix=identity_matrix[: , sorted_indices]
    # print(f"sorted_indices:\n{sorted_indices}")
    # print(f"ソートr:\n{r[sorted_indices]}")
    # print(f"permutatin:\n{permutation_matrix}")

    return permutation_matrix

# def sortPermutationMatrixByReceivedSequence(r):
    n=r.shape[0]
    identity_matrix=cp.eye(n)
    sorted_indices=cp.argsort(-cp.abs(r))
    permutation_matrix=identity_matrix[: , sorted_indices]
    # print(f"sorted_indices:\n{sorted_indices}")
    # print(f"ソートr:\n{r[sorted_indices]}")
    # print(f"permutatin:\n{permutation_matrix}")

    return permutation_matrix

#タイブレークルールとして対数事後確率比が0なら０と硬判定
def hardDecisionFromLogPosteriorProbabilityRatioS(r):
    hard_decision = cp.where(r >= 0, 0, 1)
    #おいいいい、MrrcP.C.Fossorierの本では、バイポーラ変換0->-1,1->1になってるやんけ！！！

    return hard_decision

# def hardDecisionFromReceivedSequence(r):
    hard_decision = cp.where(r >= 0, 0, 1)
    #おいいいい、MrrcP.C.Fossorierの本では、バイポーラ変換0->-1,1->1になってるやんけ！！！

    return hard_decision

def gaussian_elimination(A):
    number_of_row=A.shape[0]
    number_of_column=A.shape[1]
    U=cp.eye(number_of_row)
    P=cp.eye(number_of_column) 

    i=0
    j=0
    while i!=number_of_row:
        if (A[i][i]==1):
            #前進消去（ピボットから同列の非ゼロ成分を消す）
            indices = cp.where((A[:, i] == 1) & (cp.arange(A.shape[0]) != i))[0]
            A[indices] = (A[indices] + A[i])%2
            U[indices] = (U[indices] + U[i])%2
            i+=1
        else: #A[i][i]==0 ピボットが零である
            non_zero_position_except_for_iielement = cp.where(A[i+1:, i] == 1)[0]
            if non_zero_position_except_for_iielement.size > 0:
                position=int(non_zero_position_except_for_iielement[0] + (i + 1))
                A[[i, position ],:] = A[[position, i],:]
                U[[i, position ],:] = U[[position, i],:] 
            else:
                j+=1
                A[:,[i,i+j]]=A[:,[i+j,i]]
                P[:,[i,i+j]]=P[:,[i+j,i]]
                # print(f"j:\n{j}")

        # print(f"A:\n{A}")
        # print(f"U:\n{U}")
        # print(f"P:\n{P}")
        # print(f"i:\n{i}")
    

    # print(f"A:\n{A}")
    return U,P


def generate_combinations(array):
    n = len(array)
    ones_count = array.count(1)
    return [
        tuple(1 if i in comb else 0 for i in range(n))
        for comb in itertools.combinations(range(n), ones_count)
    ]

def OSD(u0,G,r,sigma,pi1,pi2,comb,t1):
    u0s=(comb+u0)%2
    candidate_codewords=codeword_create(G,u0s)


    candidate_codewords_pi1_pi2=cp.dot(cp.dot(candidate_codewords,pi2.T),pi1.T)
    
    # print(f"pi2:\n{pi2}")
    # print(f"candidate_codeword:\n{candidate_codewords}")
    # print(f"dot(pi2,cand):\n{cp.dot(candidate_codewords,pi2.T)}")
    # print(f"pi1:\n{pi1}")
   
    # print(f"candidate_codewords_pi1_pi2:\n{candidate_codewords_pi1_pi2}")
    candidate_codewords_pi1_pi2_bpsk=cp.where(candidate_codewords_pi1_pi2==0,1,-1)
    

    loglikehoods=MahalanobisDistancebyCholesky(r,candidate_codewords_pi1_pi2_bpsk,sigma)
    # loglikehoods=new_multi_logpdf(r,candidate_codewords_pi1_pi2_bpsk,sigma)
    # loglikehoods=MahalanobisDistances(r,candidate_codewords_pi1_pi2_bpsk,sigma)
    # loglikehoods=OnlyEuclideanDistance(r,candidate_codewords_pi1_pi2_bpsk,t1)
    
    # print(f"loglikehoods:\n{loglikehoods}")
    # print(f"r:\n{r}")
    # print(f"候補符号語:\n{candidate_codewords_pi1_pi2}")
    # print(f"尤度\n{loglikehoods}")
    # max_loglikehood=cp.min(loglikehoods)
    # max_loglikehood_index=cp.argmin(loglikehoods)
    # print(f"loglikehoods.shape:\n{loglikehoods}")
    max_loglikehood=cp.min(loglikehoods)
    max_loglikehood_index=cp.argmin(loglikehoods)
    # print(f"max_loglikehood:\n{max_loglikehood}")
    # print(f"max_loglikehood_index:\n{max_loglikehood_index}")
    codeword_estimate=candidate_codewords_pi1_pi2[max_loglikehood_index]
    if cp.isnan(max_loglikehood):
        raise ValueError("エラーが発生しました:尤度がすべてnan")
    # print(f"u0:\n{u0}")
    # print(f"u0s:\n{u0s}")
    # print(f"candidate_codewords:\n{candidate_codewords}")
    
    # print(f"codeword_estimate:\n{codeword_estimate}")
    return codeword_estimate


def CalculateMinuteValue(sigma):
    eigenvalues, _ = cp.linalg.eigh(sigma)
    min_eigenvalue=cp.min(eigenvalues)
    if min_eigenvalue > 0:
        min_eigenvalue = 0
    epsilon=cp.abs(min_eigenvalue)
    return epsilon*2.0

        

if __name__ == "__main__":
    main()