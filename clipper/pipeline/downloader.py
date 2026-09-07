import os
import yt_dlp


def download(url: str, out_dir: str = "output/raw", browser_cookies: str = None) -> str:
    """
    Download a YouTube video (best quality up to 1080p).
    Returns the path to the downloaded .mp4 file.
    """
    os.makedirs(out_dir, exist_ok=True)

    ydl_opts = {
        # Best video up to 1080p + best audio, merged to mp4
        "format": (
            "bestvideo[ext=mp4][height<=1080]+"
            "bestaudio[ext=m4a]/"
            "bestvideo[ext=mp4]+bestaudio/"
            "best[ext=mp4]/best"
        ),
        "outtmpl":              f"{out_dir}/%(id)s.%(ext)s",
        "merge_output_format":  "mp4",
        "quiet":                True,
        "no_warnings":          True,
        "noprogress":           False,
    }

    if browser_cookies:
        ydl_opts["cookiesfrombrowser"] = (browser_cookies,)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info     = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

        # Normalise extension to .mp4
        if not filename.endswith(".mp4"):
            base     = filename.rsplit(".", 1)[0]
            filename = base + ".mp4"

        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Download finished but file not found: {filename}"
            )

        size_mb = os.path.getsize(filename) / 1_048_576
        print(f"  Downloaded: {os.path.basename(filename)} ({size_mb:.1f} MB)")
        return filename
