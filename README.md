# Graduation-thesis-Program

【ファイル名】
    Graduation_Thesis_Program

【プログラム内容】
    卒業論文で検討したプログラム

【開発環境】
    numpy 2.0.2
    cupy 13.4.1
    python 3.12.6

【動作環境】
    windows11

【使い方】
    実行するとdataフォルダに2つのファイルが出力される.
    それぞれ実行条件と実行結果が表示される.
    osd.txt:実行条件と実行結果が記録されたファイル
    osd_onlydata.txt:実行結果が抽出しやすくしたファイル

    プログラムコード10行目から39行目のsetting内の変更することで符号等の条件を変更することができる
    以下setting内の変数の説明
    delta,t2:雑音成分同士の相関の大きさ,卒業論文の分散共分散行列の定義参照
    snrdB_iteration: SNR(dB)の反復回数
    snrdB_default:デフォルトのSNR(dB)値
    SNR_INTERVAL: snrdBの増加量
    word_error_iteration: シミュレーションでワード誤り率の統計を取るために繰り返すワード誤り回数
    m_range:近似周辺事後確率比における定数m, 卒業論文の近似周辺事後確率比の定義参照
    order:順序統計量復号におけるOrder
    n:符号長
    k:情報記号数
    G:生成行列


