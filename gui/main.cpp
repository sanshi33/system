#include "gui/MainWindow.h"

#include <QApplication>
#include <QFont>
#include <QMetaType>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    app.setStyle(QStringLiteral("Fusion"));
    app.setApplicationName(QStringLiteral("磨削型面误差检测系统"));
    app.setOrganizationName(QStringLiteral("Pinjie"));

    QFont uiFont;
#if defined(Q_OS_WIN)
    uiFont.setFamilies({QStringLiteral("Microsoft YaHei UI"),
                        QStringLiteral("Segoe UI"),
                        QStringLiteral("Microsoft YaHei")});
#else
    uiFont.setFamilies({QStringLiteral("Noto Sans CJK SC"),
                        QStringLiteral("Noto Sans SC"),
                        QStringLiteral("Sans Serif")});
#endif
    uiFont.setPointSize(10);
    app.setFont(uiFont);

    qRegisterMetaType<pinjie::StitchStepRecord>("pinjie::StitchStepRecord");
    qRegisterMetaType<pinjie::StitchRunCachePtr>("pinjie::StitchRunCachePtr");
    qRegisterMetaType<pinjie::CalibrationResultCache>("pinjie::CalibrationResultCache");

    pinjie::gui::MainWindow window;
    window.show();

    return app.exec();
}
