import os

from web.paths import ROOT


def main():
    os.chdir(ROOT)
    from web.app import create_app

    port = int(os.environ.get("WEB_PORT", "5050"))
    print(f"Lab app: http://127.0.0.1:{port}", flush=True)
    create_app().run(
        host="127.0.0.1",
        port=port,
        threaded=True,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
