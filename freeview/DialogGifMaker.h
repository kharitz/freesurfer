#ifndef DIALOGGIFMAKER_H
#define DIALOGGIFMAKER_H

#include <QDialog>

namespace Ui {
class DialogGifMaker;
}

class DialogGifMaker : public QDialog
{
  Q_OBJECT

public:
  explicit DialogGifMaker(QWidget *parent = nullptr);
  ~DialogGifMaker();

  void hideEvent(QHideEvent* e);

public slots:
  void OnButtonAdd();
  void OnButtonClear()
  {
    Reset();
  }
  void OnButtonSave();
  void Reset();

private:
  Ui::DialogGifMaker *ui;
  int m_nNumberOfFrames;

  void* m_gif;
  QString  m_strTempFilename;
};

#endif // DIALOGGIFMAKER_H
