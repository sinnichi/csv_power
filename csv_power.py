import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import numpy as np
from scipy.interpolate import make_interp_spline
import platform
import matplotlib.cm as cm # カラーマップをインポート

# --- Matplotlib 日本語フォント設定 ---
def set_japanese_font():
    """
    Matplotlibで日本語が正しく表示されるようにフォントを設定します。
    OSの種類に応じて適切なフォントを選択しようと試みます。
    """
    plt.rcParams['font.size'] = 12
    # マイナス記号の表示（四角になるのを防ぐ）
    plt.rcParams['axes.unicode_minus'] = False

    # OSごとのフォント設定
    if platform.system() == 'Windows':
        # Windowsでよく使われる日本語フォントをリストで指定
        plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo UI', 'Meiryo', 'MS Gothic', 'sans-serif']
    elif platform.system() == 'Darwin': # macOS
        # macOSでよく使われる日本語フォント
        plt.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS', 'sans-serif']
    else: # Linuxなど (Docker環境なども含む)
        # Linuxでよく使われる、または追加インストールされやすい日本語フォント
        # Google Noto Sans CJK JPは広く利用可能です
        plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'IPAexGothic', 'DejaVu Sans', 'sans-serif']

    # 念のため、現在設定されているフォントを確認
    # st.write(f"Matplotlib font family set to: {plt.rcParams['font.family']}")

set_japanese_font()

# --- Streamlit アプリケーションの開始 ---
st.title("📈 ハイブリッドCSV一括処理 & 出力ダッシュボード")

# CSVファイルのアップロード
uploaded_files = st.file_uploader("CSVファイルをアップロードしてください", type="csv", accept_multiple_files=True)

# 出力ディレクトリの設定（固定値）
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# デフォルト値の設定
defaults = {
    "x": None,
    "y": None,
    "kind": "line"
}
kind_default = defaults.get("kind", "line")

# グローバル変数として最小点データを保持するリスト
if 'processed_peak_data' not in st.session_state:
    st.session_state['processed_peak_data'] = {}
if 'last_uploaded_file_names' not in st.session_state:
    st.session_state['last_uploaded_file_names'] = []

# ファイルがアップロードされていない場合の処理を強化
if not uploaded_files:
    st.warning("⚠️ CSVファイルをアップロードしてください。")
    st.info("ファイルがアップロードされるのを待っています...")
else:
    current_uploaded_file_names = sorted([f.name for f in uploaded_files])
    if current_uploaded_file_names != st.session_state['last_uploaded_file_names']:
        st.session_state['processed_peak_data'] = {}
        st.session_state['last_uploaded_file_names'] = current_uploaded_file_names

    # --- CSV読み込み設定 ---
    st.sidebar.header("CSV読み込み設定")

    has_header = st.sidebar.checkbox("CSVファイルにヘッダー行がありますか？", value=True)
    st.sidebar.info("ヘッダーがある場合、CSV内の列名が表示されます。")

    skip_rows_input = st.sidebar.number_input(
        "データ開始までのスキップ行数 (0からカウント)",
        min_value=0,
        value=6,
        step=1
    )

    delimiter_options = {
        "カンマ (,)": ",",
        "タブ (\\t)": "\t",
        "スペース (複数可)": "\s+",
        "セミコロン (;)": ";"
    }
    selected_delimiter_label = st.sidebar.selectbox(
        "データ区切り文字",
        list(delimiter_options.keys()),
        index=2
    )
    selected_delimiter = delimiter_options[selected_delimiter_label]

    try:
        if has_header:
            df0 = pd.read_csv(
                BytesIO(uploaded_files[0].getvalue()),
                skiprows=skip_rows_input,
                nrows=1,
                sep=selected_delimiter,
                engine='python'
            )
            x_col_options = df0.columns.tolist()
            y_col_options = df0.columns.tolist()
        else:
            df0 = pd.read_csv(
                BytesIO(uploaded_files[0].getvalue()),
                header=None,
                skiprows=skip_rows_input,
                nrows=5,
                sep=selected_delimiter,
                engine='python'
            )
            x_col_options = [chr(65 + i) if i < 26 else str(i) for i in range(len(df0.columns))]
            y_col_options = [chr(65 + i) if i < 26 else str(i) for i in range(len(df0.columns))]
            st.sidebar.info("ヘッダーなしを選択した場合、列はA, B, C...または数値で表示されます。")

    except Exception as e:
        st.error(f"CSVファイルの初期読み込みエラーが発生しました。設定を確認してください。エラー詳細: {e}")
        st.warning("考えられる原因: スキップ行数、または区切り文字が間違っている可能性があります。")
        st.stop()

    default_x_col = x_col_options[0] if x_col_options else None
    default_y_col = x_col_options[1] if len(x_col_options) > 1 else (x_col_options[0] if x_col_options else None)

    st.sidebar.header("グラフ描画設定")

    x_col_default_index = 0
    if default_x_col and default_x_col in x_col_options:
        x_col_default_index = x_col_options.index(default_x_col)

    y_col_default_index = 0
    if default_y_col and default_y_col in y_col_options:
        y_col_default_index = y_col_options.index(default_y_col)

    x_col_selected = st.selectbox("X軸の列", x_col_options, index=x_col_default_index, key="x_select")
    y_col_selected = st.selectbox("Y軸の列", y_col_options, index=y_col_default_index, key="y_select")
    kind = st.radio("グラフ種類", ["line", "scatter", "bar"], index=["line","scatter","bar"].index(kind_default))

    # --- X軸オフセット設定 ---
    st.sidebar.subheader("X軸オフセット設定")
    x_offset_value = st.sidebar.number_input(
        "X軸オフセット値",
        value=0.0,
        step=0.0001, # 精度を向上
        format="%.4f", # 表示精度を向上
        help="グラフのX軸に加算される値を設定します。例: -712700000"
    )

    # --- 新機能: 軸範囲のカスタム設定 ---
    st.sidebar.subheader("軸範囲のカスタム設定")
    use_custom_x_lim = st.sidebar.checkbox("X軸範囲をカスタム設定", value=False)
    x_lim_min = None
    x_lim_max = None
    if use_custom_x_lim:
        x_lim_min = st.sidebar.number_input("X軸 最小値", value=0.0, format="%.4f", step=0.0001, key="x_lim_min") # 精度を向上
        x_lim_max = st.sidebar.number_input("X軸 最大値", value=100.0, format="%.4f", step=0.0001, key="x_lim_max") # 精度を向上

    use_custom_y_lim = st.sidebar.checkbox("Y軸範囲をカスタム設定", value=False)
    y_lim_min = None
    y_lim_max = None
    if use_custom_y_lim:
        y_lim_min = st.sidebar.number_input("Y軸 最小値", value=-100.0, format="%.2f", key="y_lim_min")
        y_lim_max = st.sidebar.number_input("Y軸 最大値", value=0.0, format="%.2f", key="y_lim_max")

    # --- 新機能: 線の太さ設定 ---
    st.sidebar.subheader("線の見易さ設定")
    line_width = st.sidebar.number_input("線の太さ", value=1.5, min_value=0.5, max_value=5.0, step=0.1)

    # --- 新機能: カスタム軸ラベル設定 ---
    st.sidebar.subheader("軸ラベル設定")
    custom_x_label = st.sidebar.text_input("カスタムX軸ラベル (空欄でデフォルト)", value="")
    custom_y_label = st.sidebar.text_input("カスタムY軸ラベル (空欄でデフォルト)", value="")


    def plot_and_save(file_object, x_col_name, y_col_name, kind, save_path, has_header_plot, skip_rows_plot, delimiter_plot, x_offset, x_lim_min, x_lim_max, y_lim_min, y_lim_max, line_width, custom_x_label_plot, custom_y_label_plot):
        """
        単一のCSVファイルを読み込み、指定された設定でグラフをプロットし、保存します。
        """
        try:
            if has_header_plot:
                df = pd.read_csv(
                    BytesIO(file_object.getvalue()),
                    skiprows=skip_rows_plot,
                    sep=delimiter_plot,
                    engine='python'
                )
            else:
                df = pd.read_csv(
                    BytesIO(file_object.getvalue()),
                    header=None,
                    skiprows=skip_rows_plot,
                    sep=delimiter_plot,
                    engine='python'
                )

            if x_col_name not in df.columns:
                st.error(f"エラー: ファイル '{file_object.name}' にX軸の列 '{x_col_name}' が見つかりません。")
                return
            if y_col_name not in df.columns:
                st.error(f"エラー: ファイル '{file_object.name}' にY軸の列 '{y_col_name}' が見つかりません。")
                return

            df[x_col_name] = pd.to_numeric(df[x_col_name], errors='coerce')
            df[y_col_name] = pd.to_numeric(df[y_col_name], errors='coerce')
            df.dropna(subset=[x_col_name, y_col_name], inplace=True)

            if df.empty:
                st.warning(f"ファイル '{file_object.name}' の選択された列に有効なデータがありません。グラフは生成されません。")
                return

            plt.figure(figsize=(8,5))

            # X軸にオフセットを適用
            df_plot = df.copy()
            df_plot[x_col_name] = df_plot[x_col_name] + x_offset

            if kind == 'line':
                df_sorted = df_plot.sort_values(by=x_col_name).copy()
                x_data = df_sorted[x_col_name].values
                y_data = df_sorted[y_col_name].values

                if len(df_sorted) > 3000:
                    sampling_rate = max(1, len(df_sorted) // 3000)
                    df_for_plot_sampled = df_sorted.iloc[::sampling_rate, :].copy()
                    x_data = df_for_plot_sampled[x_col_name].values
                    y_data = df_for_plot_sampled[y_col_name].values

                if len(set(x_data)) < len(x_data) or len(x_data) <= 1:
                    plt.plot(x_data, y_data, linewidth=line_width)
                else:
                    try:
                        if len(x_data) > 3:
                            x_new = np.linspace(x_data.min(), x_data.max(), 500)
                            spl = make_interp_spline(x_data, y_data, k=min(3, len(x_data) - 1))
                            y_smooth = spl(x_new)
                            plt.plot(x_new, y_smooth, linewidth=line_width)
                        else:
                            plt.plot(x_data, y_data, linewidth=line_width)
                    except Exception as e:
                        st.warning(f"ファイル '{file_object.name}' のラインプロットで補間エラー: {e}。生データをプロットします。")
                        plt.plot(x_data, y_data, linewidth=line_width)

            elif kind == "scatter":
                if len(df_plot) > 3000:
                    sampling_rate = max(1, len(df_plot) // 3000)
                    df_for_plot_sampled = df_plot.iloc[::sampling_rate, :].copy()
                    plt.scatter(df_for_plot_sampled[x_col_name], df_for_plot_sampled[y_col_name], s=5)
                else:
                    plt.scatter(df_plot[x_col_name], df_plot[y_col_name], s=5)
            elif kind == "bar":
                plt.bar(df_plot[x_col_name], df_plot[y_col_name])

            # --- グラフの軸ラベルとタイトル ---
            plt.xlabel(custom_x_label_plot if custom_x_label_plot else x_col_name) # カスタムラベルを適用
            plt.ylabel(custom_y_label_plot if custom_y_label_plot else y_col_name) # カスタムラベルを適用
            plt.title(f"{file_object.name.split('.')[0]} ({y_col_name} vs {x_col_name})")

            peak_info_display = None
            min_x_at_min_y = None
            min_y_value = None

            if y_col_name == "S21(DB)":
                min_y_value_original_df = df[y_col_name].min()
                min_y_row_original_df = df[df[y_col_name] == min_y_value_original_df].iloc[0]
                min_x_at_min_y_original = min_y_row_original_df[x_col_name]

                min_x_at_min_y_displayed = min_x_at_min_y_original + x_offset

                plt.annotate(
                    f'Min: X={min_x_at_min_y_displayed:.2f}, Y={min_y_value_original_df:.4f}',
                    xy=(min_x_at_min_y_displayed, min_y_value_original_df),
                    xytext=(min_x_at_min_y_displayed, min_y_value_original_df),
                    textcoords="offset points",
                    xycoords='data',
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    fontsize=9,
                    color='red'
                )
                peak_info_display = f"ファイル: {file_object.name.split('.')[0]} - 最小点: X={min_x_at_min_y_displayed:.2f}, Y={min_y_value_original_df:.4f}"

                min_x_coord_for_session = min_x_at_min_y_original

            # 軸範囲を適用
            if x_lim_min is not None and x_lim_max is not None:
                plt.xlim(x_lim_min, x_lim_max)
            if y_lim_min is not None and y_lim_max is not None:
                plt.ylim(y_lim_min, y_lim_max)

            plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_path, format='png', dpi=300) # formatとdpiを明示

            plt.close()

            return peak_info_display, file_object.name, min_x_coord_for_session, min_y_value_original_df

        except Exception as e:
            st.error(f"ファイル '{file_object.name}' のグラフ生成中にエラーが発生しました。詳細: {e}")
            plt.close()
            return None, None, None, None

    # --- 新しい関数: 複数のファイルを重ねてプロット ---
    def plot_overlay_and_save(uploaded_files, x_col_name, y_col_name, kind, output_dir, has_header_plot, skip_rows_plot, delimiter_plot, x_offset, x_lim_min, x_lim_max, y_lim_min, y_lim_max, line_width, custom_x_label_plot, custom_y_label_plot):
        plt.figure(figsize=(12, 7))
        # ファイル数に応じて異なる色を生成 (tab10は最大10色)
        colors = cm.get_cmap('tab10', max(len(uploaded_files), 10)).colors

        for i, file_object in enumerate(uploaded_files):
            try:
                if has_header_plot:
                    df = pd.read_csv(
                        BytesIO(file_object.getvalue()),
                        skiprows=skip_rows_plot,
                        sep=delimiter_plot,
                        engine='python'
                    )
                else:
                    df = pd.read_csv(
                        BytesIO(file_object.getvalue()),
                        header=None,
                        skiprows=skip_rows_plot,
                        sep=delimiter_plot,
                        engine='python'
                    )

                if x_col_name not in df.columns:
                    st.warning(f"警告: ファイル '{file_object.name}' にX軸の列 '{x_col_name}' が見つかりません。このファイルはスキップされます。")
                    continue
                if y_col_name not in df.columns:
                    st.warning(f"警告: ファイル '{file_object.name}' にY軸の列 '{y_col_name}' が見つかりません。このファイルはスキップされます。")
                    continue

                df[x_col_name] = pd.to_numeric(df[x_col_name], errors='coerce')
                df[y_col_name] = pd.to_numeric(df[y_col_name], errors='coerce')
                df.dropna(subset=[x_col_name, y_col_name], inplace=True)

                if df.empty:
                    st.warning(f"ファイル '{file_object.name}' の選択された列に有効なデータがありません。このファイルはスキップされます。")
                    continue

                df_plot = df.copy()
                df_plot[x_col_name] = df_plot[x_col_name] + x_offset

                label = file_object.name.split('.')[0] # ファイル名をラベルとして使用

                if kind == 'line':
                    df_sorted = df_plot.sort_values(by=x_col_name).copy()
                    x_data = df_sorted[x_col_name].values
                    y_data = df_sorted[y_col_name].values

                    if len(df_sorted) > 3000:
                        sampling_rate = max(1, len(df_sorted) // 3000)
                        df_for_plot_sampled = df_sorted.iloc[::sampling_rate, :].copy()
                        x_data = df_for_plot_sampled[x_col_name].values
                        y_data = df_for_plot_sampled[y_col_name].values

                    if len(set(x_data)) < len(x_data) or len(x_data) <= 1:
                        plt.plot(x_data, y_data, label=label, color=colors[i % len(colors)], linewidth=line_width)
                    else:
                        try:
                            if len(x_data) > 3:
                                x_new = np.linspace(x_data.min(), x_data.max(), 500)
                                spl = make_interp_spline(x_data, y_data, k=min(3, len(x_data) - 1))
                                y_smooth = spl(x_new)
                                plt.plot(x_new, y_smooth, label=label, color=colors[i % len(colors)], linewidth=line_width)
                            else:
                                plt.plot(x_data, y_data, label=label, color=colors[i % len(colors)], linewidth=line_width)
                        except Exception as e:
                            st.warning(f"ファイル '{file_object.name}' のラインプロットで補間エラー: {e}。生データをプロットします。")
                            plt.plot(x_data, y_data, label=label, color=colors[i % len(colors)], linewidth=line_width)

                elif kind == "scatter":
                    if len(df_plot) > 3000:
                        sampling_rate = max(1, len(df_plot) // 3000)
                        df_for_plot_sampled = df_plot.iloc[::sampling_rate, :].copy()
                        plt.scatter(df_for_plot_sampled[x_col_name], df_for_plot_sampled[y_col_name], s=5, label=label, color=colors[i % len(colors)])
                    else:
                        plt.scatter(df_plot[x_col_name], df_plot[y_col_name], s=5, label=label, color=colors[i % len(colors)])
                elif kind == "bar":
                    plt.bar(df_plot[x_col_name], df_plot[y_col_name], label=label, color=colors[i % len(colors)])

            except Exception as e:
                st.error(f"ファイル '{file_object.name}' の重ね合わせグラフ生成中にエラーが発生しました。詳細: {e}")
                continue

        plt.xlabel(custom_x_label_plot if custom_x_label_plot else x_col_name) # カスタムラベルを適用
        plt.ylabel(custom_y_label_plot if custom_y_label_plot else y_col_name) # カスタムラベルを適用
        plt.title(f"全ファイルの重ね合わせ ({y_col_name} vs {x_col_name})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) # 凡例を外に出す
        plt.grid(True)
        plt.tight_layout()

        # 軸範囲を適用
        if x_lim_min is not None and x_lim_max is not None:
            plt.xlim(x_lim_min, x_lim_max)
        if y_lim_min is not None and y_lim_max is not None:
            plt.ylim(y_lim_min, y_lim_max)

        overlay_plot_filename = f"combined_overlay_plot_{y_col_name}_vs_{x_col_name}.png"
        overlay_plot_path = os.path.join(output_dir, overlay_plot_filename)
        plt.savefig(overlay_plot_path, format='png', dpi=300)
        plt.close()
        return overlay_plot_path


    if st.button("全ファイル一括処理"):
        st.info("⚙️ 並列で処理中…")
        st.session_state['processed_peak_data'] = {}

        max_workers_to_use = 1

        with ThreadPoolExecutor(max_workers=max_workers_to_use) as executor:
            futures = []
            for uploaded_file in uploaded_files:
                fname = os.path.splitext(uploaded_file.name)[0]
                save_name = f"{fname}_{y_col_selected}_vs_{x_col_selected}.png"
                save_path = os.path.join(output_dir, save_name)
                futures.append(
                    executor.submit(
                        plot_and_save, uploaded_file, x_col_selected, y_col_selected, kind,
                        save_path,
                        has_header, skip_rows_input, selected_delimiter, x_offset_value,
                        x_lim_min, x_lim_max, y_lim_min, y_lim_max, line_width, # 新しい引数
                        custom_x_label, custom_y_label # 新しい引数
                    )
                )

            for future in futures:
                display_info, file_name, x_coord, y_min_val = future.result()
                if x_coord is not None:
                    st.session_state['processed_peak_data'][file_name] = {
                        'x_coord': x_coord,
                        'display_info': display_info,
                        'y_min_val': y_min_val
                    }
        st.success("✅ 全ファイル処理完了！")

        if st.session_state['processed_peak_data']:
            st.header("📉 各グラフの最小点座標")
            for file_name, data in st.session_state['processed_peak_data'].items():
                if data['display_info']:
                    st.write(data['display_info'])

    # --- 新しいボタン: 全ファイルを重ねてプロット ---
    st.header("✨ 全ファイルを重ねてプロット")
    if st.button("全ファイルを重ねてプロット"):
        if uploaded_files:
            st.info("⚙️ 重ね合わせグラフを生成中…")
            overlay_plot_path = plot_overlay_and_save(
                uploaded_files, x_col_selected, y_col_selected, kind, output_dir,
                has_header, skip_rows_input, selected_delimiter, x_offset_value,
                x_lim_min, x_lim_max, y_lim_min, y_lim_max, line_width, # 新しい引数
                custom_x_label, custom_y_label # 新しい引数
            )
            if os.path.exists(overlay_plot_path):
                st.success(f"重ね合わせグラフ '{os.path.basename(overlay_plot_path)}' が生成されました！")
                with open(overlay_plot_path, "rb") as img_file:
                    img_bytes_for_download = img_file.read()
                    st.image(img_bytes_for_download, caption=os.path.basename(overlay_plot_path), width=600)
                    st.download_button("⬇️ ダウンロード", img_bytes_for_download, file_name=os.path.basename(overlay_plot_path), mime="image/png", key="download_overlay_plot")
            else:
                st.error("重ね合わせグラフの生成に失敗しました。")
        else:
            st.warning("重ね合わせプロットにはCSVファイルのアップロードが必要です。")


    st.header("📂 出力ファイル一覧")

    output_files = [f for f in os.listdir(output_dir) if f.endswith(".png")]
    if not output_files:
        st.info("出力ファイルはまだありません。")
    else:
        for f in output_files:
            path = os.path.join(output_dir, f)
            if os.path.exists(path):
                with open(path, "rb") as img_file:
                    img_bytes = img_file.read()
                    st.image(img_bytes, caption=f, width=300)
                    st.download_button("⬇️ ダウンロード", img_bytes, file_name=f, mime="image/png", key=f"download_plot_{f}")
            else:
                st.warning(f"画像ファイル '{f}' が見つかりません。")

    # --- 新機能: ファイル削除セクション ---
    st.header("🗑️ 出力ファイル管理")
    manage_output_files = [f for f in os.listdir(output_dir) if f.endswith(".png")]
    if not manage_output_files:
        st.info("削除できる出力ファイルはありません。")
    else:
        st.write("削除したいファイルを選択してください:")
        for f in manage_output_files:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.write(f)
            with col2:
                if st.button("削除", key=f"delete_file_{f}"):
                    file_to_delete = os.path.join(output_dir, f)
                    try:
                        os.remove(file_to_delete)
                        st.success(f"ファイル '{f}' を削除しました。")
                        st.rerun() # ファイルリストを更新するために再実行
                    except OSError as e:
                        st.error(f"ファイル '{f}' の削除中にエラーが発生しました: {e}")

    # --- 新機能: 複数の最小点X座標と各点ごとの指定デシベル値でのプロット ---
    st.header("📊 最小点X座標と各ファイル指定デシベル値でのプロット")

    if st.session_state['processed_peak_data']:
        st.write("処理されたファイルの最小点X座標と、設定可能なY値:")

        plot_x_values_new_plot = []
        plot_y_values_new_plot = []
        plot_labels_new_plot = []

        cols = st.columns([0.6, 0.4])
        with cols[0]:
            st.write("**ファイル名 (最小点X)**")
        with cols[1]:
            st.write("**Y軸 (dB)**")

        for uploaded_file_obj in uploaded_files:
            file_name = uploaded_file_obj.name
            if file_name in st.session_state['processed_peak_data']:
                peak_data = st.session_state['processed_peak_data'][file_name]
                x_val_original = peak_data['x_coord']

                input_key = f"{file_name}_y_input"

                # session_stateにこのキーがまだ存在しない場合のみ初期値を設定
                if input_key not in st.session_state:
                    st.session_state[input_key] = float(peak_data['y_min_val'] if peak_data['y_min_val'] is not None else -20.0)

                with st.container():
                    col1, col2 = st.columns([0.6, 0.4])
                    with col1:
                        x_val_displayed_for_new_plot = x_val_original + x_offset_value
                        st.write(f"**{file_name.split('.')[0]}** (X={x_val_displayed_for_new_plot:.4f})") # 表示精度を向上
                    with col2:
                        st.number_input(
                            "",
                            value=st.session_state[input_key],
                            step=0.1,
                            format="%.2f",
                            key=input_key
                        )

                if x_val_original is not None:
                    plot_x_values_new_plot.append(x_val_original + x_offset_value)
                    plot_y_values_new_plot.append(st.session_state[input_key])
                    plot_labels_new_plot.append(file_name.split('.')[0])

        if st.button("選択Y値で最小点X座標をプロット"):
            if plot_x_values_new_plot:
                plt.figure(figsize=(10, 6))

                plt.scatter(plot_x_values_new_plot, plot_y_values_new_plot, color='blue', s=50, zorder=2)

                for i, x_val in enumerate(plot_x_values_new_plot):
                    plt.annotate(
                        plot_labels_new_plot[i],
                        (x_val, plot_y_values_new_plot[i]),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=8
                    )

                x_label_for_new_plot = f"最小点のX座標 ({x_col_selected}"
                if x_offset_value != 0:
                    x_label_for_new_plot += f" + {x_offset_value:.4f})" # 表示精度を向上
                else:
                    x_label_for_new_plot += ")"

                plt.xlabel(custom_x_label if custom_x_label else x_label_for_new_plot) # カスタムラベルを適用
                plt.ylabel(custom_y_label if custom_y_label else "指定デシベル値 (dB)") # カスタムラベルを適用
                plt.title(f"複数のファイルの最小点X座標と指定デシベル値")
                plt.grid(True)
                plt.tight_layout()

                # 軸範囲を適用 (このプロットにも適用できるように)
                if x_lim_min is not None and x_lim_max is not None:
                    plt.xlim(x_lim_min, x_lim_max)
                if y_lim_min is not None and y_lim_max is not None:
                    plt.ylim(y_lim_min, y_lim_max)

                new_plot_filename = f"min_x_coords_per_file_plot.png"
                new_plot_path = os.path.join(output_dir, new_plot_filename)
                plt.savefig(new_plot_path, format='png', dpi=300)
                plt.close()

                st.success(f"新しいグラフ '{new_plot_filename}' が生成されました！")
                if os.path.exists(new_plot_path):
                    with open(new_plot_path, "rb") as img_file:
                        img_bytes_for_download = img_file.read()
                        st.image(img_bytes_for_download, caption=new_plot_filename, width=400)
                        st.download_button("⬇️ ダウンロード", img_bytes_for_download, file_name=new_plot_filename, mime="image/png", key="download_min_x_plot")
                else:
                    st.error(f"エラー: 生成された画像ファイル '{new_plot_filename}' が見つかりません。")
            else:
                st.warning("プロットするデータがありません。まず「全ファイル一括処理」を実行し、最小点X座標が検出されていることを確認してください。")
    else:
        st.info("最小点X座標データを収集するには、まずCSVファイルをアップロードし、「全ファイル一括処理」を実行してください。")