# Windows での導入と配布

> **確認状況**: この手順は Linux 上で作成・検証しました。PyInstaller による配布フォルダの作成と、
> その自己診断（`--selftest`）が通ることは Linux で確認済みです。**Windows 実機でのビルド・起動は未確認**です。
> 初回は下の「確認チェックリスト」で実機確認をお願いします。

## 前提（仮置き）

- Windows 10 / 11 の 64bit。管理者権限は不要の想定（配布フォルダを利用者の書込み可能な場所に置く場合）。
- ビルドを行う PC に Python 3.11 以降の 64bit（開発・試験は 3.13）。インストール時に「Add python.exe to PATH」を有効にする。
- ビルド時だけネットワークが必要（ライブラリの取得）。**配布先の PC は不要・オフラインで動作**します。

## A. ソースから起動する（開発・試験用）

1. このリポジトリのフォルダでコマンドプロンプトを開く。
2. `pip install -r requirements.txt`
3. `run_windows.bat`（または `python app.py`）。

## B. 配布用の exe を作る

1. ビルド用 PC で `build_windows.bat` をダブルクリック（またはコマンドプロンプトで実行）。
   - 仮想環境の作成 → ライブラリ導入 → 自己診断 → exe 作成 → exe の自己診断、の順に進みます。
2. 完成物は `dist\TraceView\` フォルダです。**フォルダごと**（zip などで）配布します。`TraceView.exe` を起動すると使えます。
3. 配布先では Python のインストールは不要です。

## 確認チェックリスト（Windows 実機）

| # | 確認 | 期待 |
|---|---|---|
| 1 | `python app.py --selftest` | 「すべて正常です」、`selftest_result.txt` に `RESULT: OK` |
| 2 | `build_windows.bat` | 最後に exe の自己診断が `RESULT: OK` |
| 3 | 配布先PCで `TraceView.exe` を起動 | 画面が開く（Python未導入のPCで） |
| 4 | **ネットワークを切断**して、PDFを開く→自動解析→Excel出力 | 最後まで動く |
| 5 | 出力した .xlsx を Excel で開く | 3シート＋画像が表示され、日本語が文字化けしない |
| 6 | 日本語のフォルダ名・ファイル名のPDFを開く | 開ける |

## 困ったとき

| 症状 | 対処 |
|---|---|
| `python` が見つからない | Python を再インストールし「Add python.exe to PATH」を有効にする |
| `pip install` が失敗 | ネットワーク／プロキシを確認。社内のプロキシ設定が必要な場合は管理者に確認 |
| exe がウイルス対策ソフトに止められる | PyInstaller 製 exe は誤検知されることがあります。社内のセキュリティ担当に確認し、許可リストへ登録 |
| 起動しても何も出ない | `run_windows.bat` から起動してメッセージを確認。`TraceView.exe --selftest` の結果は同じフォルダの `selftest_result.txt` |
| Excel出力が失敗する | 同名ファイルを Excel で開いている場合は閉じる。出力先の書込み権限を確認 |

## サイズについて

配布フォルダは、PySide6 などを同梱するため大きくなります。Linux の確認では約600MBでした
（この環境には使用しない Qt モジュールも入っていたため）。Windows では `requirements.txt` が
`PySide6-Essentials` を使うため、これより小さくなる見込みですが、実機での大きさは未確認です。
