declare const _default: {
    content: string[];
    theme: {
        extend: {
            fontFamily: {
                display: [string, string, string];
                body: [string, string, string];
            };
            colors: {
                backdrop: string;
                panel: string;
                accent: string;
                accentSoft: string;
            };
            boxShadow: {
                glass: string;
            };
            animation: {
                drift: string;
            };
            keyframes: {
                drift: {
                    "0%, 100%": {
                        transform: string;
                    };
                    "50%": {
                        transform: string;
                    };
                };
            };
        };
    };
    plugins: any[];
};
export default _default;
