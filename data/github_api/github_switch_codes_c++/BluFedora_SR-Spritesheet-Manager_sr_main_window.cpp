//
// SR Texture Packer
// Copyright (c) 2020 Shareef Aboudl-Raheem
//

#include "sr_main_window.hpp"

#include "Data/sr_settings.hpp"
#include "UI/sr_timeline.hpp"
#include "UI/sr_welcome_window.hpp"

#include "Server/sr_live_reload_server.hpp"  // g_Server
#include "sr_new_animation_dialog.hpp"

#include <QCloseEvent>
#include <QFileDialog>
#include <QInputDialog>
#include <QMessageBox>

MainWindow::MainWindow(const QString& name, QWidget* parent) :
  QMainWindow(parent),
  m_BaseTitle{},
  m_OpenProject{std::make_unique<Project>(this, name)},
  m_PacketSendingProgress{this},
  m_OnTimelineChange{}
{
  setupUi(this);

  m_BaseTitle = windowTitle();

  setWindowModified(!m_OpenProject->hasPath());
  setTabPosition(Qt::AllDockWidgetAreas, QTabWidget::North);
  setCorner(Qt::BottomLeftCorner, Qt::LeftDockWidgetArea);
  setCorner(Qt::BottomRightCorner, Qt::RightDockWidgetArea);

  m_UndoView->setStack(&m_OpenProject->historyStack());
  m_AnimationList->setModel(&m_OpenProject->animations());
  m_OpenProject->setup(m_ImageLibrary);
  m_TimelineFrames->setup(m_Timeline);

  m_OnTimelineChange.slider = m_TimelineFrameSizeSlider;

  QObject::connect(m_TimelineFrameSizeSlider, &QSlider::valueChanged, m_TimelineFrames, &Timeline::onFrameSizeChanged);
  QObject::connect(m_TimelineFpsSpinbox, &QSpinBox::valueChanged, m_OpenProject.get(), &Project::onTimelineFpsChange);
  QObject::connect(m_OpenProject.get(), &Project::atlasModified, m_TimelineFrames, &Timeline::onAtlasUpdated);
  QObject::connect(m_OpenProject.get(), &Project::atlasModified, m_GfxPreview, &AnimationPreview::onAtlasUpdated);
  QObject::connect(m_OpenProject.get(), &Project::animationChanged, m_TimelineFrames, &Timeline::onAnimationChanged);
  QObject::connect(m_OpenProject.get(), &Project::animationChanged, this, &MainWindow::onAnimChanged);
  QObject::connect(m_OpenProject.get(), &Project::animationChanged, m_GfxPreview, &AnimationPreview::onAnimationChanged);
  QObject::connect(m_OpenProject.get(), &Project::animationSelected, m_TimelineFrames, &Timeline::onAnimationSelected);
  QObject::connect(m_OpenProject.get(), &Project::animationSelected, m_GfxPreview, &AnimationPreview::onAnimationSelected);
  QObject::connect(m_OpenProject.get(), &Project::signalPreviewFrameSelected, m_GfxPreview, &AnimationPreview::onFrameSelected);
  QObject::connect(m_ActionImageLibraryNewFolder, &QAction::triggered, m_OpenProject.get(), &Project::onCreateNewFolder);
  QObject::connect(m_ActionImportImages, &QAction::triggered, m_OpenProject.get(), &Project::onImportImages);
  QObject::connect(m_OpenProject.get(), &Project::renamed, this, &MainWindow::onProjectRenamed);
  QObject::connect(m_AnimationList, &QListView::customContextMenuRequested, this, &MainWindow::onAnimationRightClick);
  QObject::connect(m_AnimationList->selectionModel(), &QItemSelectionModel::currentRowChanged, this, &MainWindow::onAnimationSelectionChanged);

  m_TimelineFrames->frameHeight().addListener(&m_OnTimelineChange);

  auto& undo_stack = m_OpenProject->historyStack();

  auto edit_menu   = new QMenu(tr("&Edit"), this);
  auto undo_action = undo_stack.createUndoAction(this, tr("&Undo"));
  auto redo_action = undo_stack.createRedoAction(this, tr("&Redo"));

  QMainWindow::menuBar()->insertMenu(menuImageLibrary->menuAction(), edit_menu);

  auto window_menu = new QMenu(tr("&Window"), this);

  window_menu->addAction(m_DockImageLibraryView->toggleViewAction());
  window_menu->addAction(m_DockTimelineView->toggleViewAction());
  window_menu->addAction(m_AnimationListDock->toggleViewAction());
  window_menu->addAction(m_DockPropertyView->toggleViewAction());
  window_menu->addAction(m_DockHistoryView->toggleViewAction());

  QMainWindow::menuBar()->insertMenu(menuAbout->menuAction(), window_menu);

  undo_action->setShortcuts(QKeySequence::Undo);
  redo_action->setShortcuts(QKeySequence::Redo);

  edit_menu->addAction(undo_action);
  edit_menu->addAction(redo_action);

  m_TimelinePlayButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));

  QObject::connect(m_TimelinePlayButton, &QToolButton::clicked, [this]() {
    m_GfxPreview->onTogglePlayAnimation();
    m_TimelinePlayButton->setIcon(m_GfxPreview->isPlayingAnimation() ? style()->standardIcon(QStyle::SP_MediaPause) : style()->standardIcon(QStyle::SP_MediaPlay));
  });

  if (g_Server)
  {
    m_StatusBar->showMessage("Data => Engine");
    m_StatusBar->addPermanentWidget(&m_PacketSendingProgress);

    QObject::connect(g_Server.get(), &LiveReloadServer::bytesSent, [this]() {
      m_PacketSendingProgress.setRange(0, g_Server->numBytesNeedSend());
      m_PacketSendingProgress.setValue(g_Server->numBytesSent());
    });
  }

  restoreWindowLayout();
}

void MainWindow::postLoadInit()
{
  onProjectRenamed(m_OpenProject->name());
  m_QualitySpritesheetSize->setValue(m_OpenProject->spritesheetImageSize());
  m_QualityFrameSize->setValue(m_OpenProject->spritesheetFrameSize());

  QObject::connect(m_QualitySpritesheetSize, &QSpinBox::editingFinished, this, &MainWindow::onSpritesheetQualitySettingChanged);
  QObject::connect(m_QualityFrameSize, &QSpinBox::editingFinished, this, &MainWindow::onSpritesheetQualitySettingChanged);
}

void MainWindow::onProjectRenamed(const QString& name)
{
  setWindowTitle(name + m_BaseTitle);
}

void MainWindow::onSaveProject()
{
  m_OpenProject->save();
}

void MainWindow::onAnimationNew()
{
  NewAnimation new_anim_dlog;

  if (new_anim_dlog.exec() == QDialog::Accepted)
  {
    m_OpenProject->newAnimation(new_anim_dlog.name(), new_anim_dlog.frameRate());
  }
}

void MainWindow::onAnimationRightClick(const QPoint& pos)
{
  const QModelIndexList selected_items = m_AnimationList->selectionModel()->selectedIndexes();

  if (!selected_items.isEmpty())
  {
    QMenu r_click;

    auto* const rename = r_click.addAction("Rename Animation");
    auto* const remove = r_click.addAction("Remove Animation");
    const auto  result = r_click.exec(m_AnimationList->mapToGlobal(pos));

    if (result == remove)
    {
      m_OpenProject->removeAnimation(selected_items[0]);
    }

    if (result == rename)
    {
      for (const auto& item : selected_items)
      {
        Animation* const animation = m_OpenProject->animationAt(item);
        const auto       name      = QInputDialog::getText(this, "Animation", "NAME: ", QLineEdit::Normal, animation->name());

        m_OpenProject->setAnimationName(animation, name);

        break;
      }
    }
  }
}

void MainWindow::onShowWelcomeScreen()
{
  (new WelcomeWindow())->show();
}

void MainWindow::onSpritesheetQualitySettingChanged()
{
  m_OpenProject->setSpritesheetImageSize(m_QualitySpritesheetSize->value());
  m_OpenProject->setSpritesheetFrameSize(m_QualityFrameSize->value());
}

void MainWindow::onAnimationSelectionChanged(const QModelIndex& current, const QModelIndex& /* previous */)
{
  m_OpenProject->selectAnimation(current);
  onAnimChanged(m_OpenProject->selectedAnimation());
}

void MainWindow::onAnimChanged(Animation* anim)
{
  if (anim)
  {
    m_MiniScrubber->setRange(0, anim->numFrames() - 1);
  }
}

void MainWindow::changeEvent(QEvent* e)
{
  QMainWindow::changeEvent(e);

  switch (e->type())
  {
    case QEvent::LanguageChange:
      retranslateUi(this);
      break;
    default:
      break;
  }
}

void MainWindow::on_m_ActionAboutQt_triggered()
{
  QMessageBox::aboutQt(this, "About Qt");
}

void MainWindow::closeEvent(QCloseEvent* event)
{
  Settings settings;

  settings.setValue("MainWindow/geometry", saveGeometry());
  settings.setValue("MainWindow/windowState", saveState());
  settings.setValue("MainWindow/isMaximized", isMaximized());

  if (isWindowModified())
  {
    const QMessageBox::StandardButton res_btn = QMessageBox::question(this, windowTitle(), tr("You have unsaved changes would you like to save?\n"), QMessageBox::Cancel | QMessageBox::No | QMessageBox::Yes, QMessageBox::Yes);

    if (res_btn == QMessageBox::Yes)
    {
      if (m_OpenProject->save())
      {
        event->accept();
      }
      else
      {
        QMessageBox::warning(this, "Warning", "Failed to save project.", QMessageBox::Ok, QMessageBox::Ok);
        event->ignore();
      }
    }
    else if (res_btn == QMessageBox::Cancel)
    {
      event->ignore();
    }
    else if (res_btn == QMessageBox::No)
    {
      event->accept();
    }
    else
    {
      event->ignore();
    }
  }
  else
  {
    event->accept();
  }
}

void MainWindow::restoreWindowLayout()
{
  Settings settings;

  restoreState(settings.value("MainWindow/windowState").toByteArray());
  if (settings.value("MainWindow/isMaximized").toBool())
  {
    setWindowState(Qt::WindowMaximized);
  }
  else
  {
    restoreGeometry(settings.value("MainWindow/geometry").toByteArray());
  }

  restoreDockWidget(m_DockImageLibraryView);
  restoreDockWidget(m_DockTimelineView);
  restoreDockWidget(m_AnimationListDock);
  restoreDockWidget(m_DockPropertyView);
  restoreDockWidget(m_DockHistoryView);
}

void MainWindow::on_m_ActionExportSpritesheet_triggered()
{
  const QString export_dir = QFileDialog::getExistingDirectory(this, "Export Location", QString());

  if (!export_dir.isEmpty())
  {
    if (!m_OpenProject->exportAtlas(export_dir))
    {
      QMessageBox::warning(this, "Warning", "Failed to export spritesheet.", QMessageBox::Ok, QMessageBox::Ok);
    }
  }
}

void MainWindow::on_m_ActionProjectRename_triggered()
{
  const QString new_name = QInputDialog::getText(this, tr("Rename Project"), "New Name", QLineEdit::Normal, m_OpenProject->name());

  if (!new_name.isEmpty())
  {
    m_OpenProject->setProjectName(new_name);
  }
}

void MainWindow::on_m_MiniScrubber_sliderMoved(int position)
{
  Animation* const anim = m_OpenProject->selectedAnimation();

  if (anim)
  {
    anim->previewed_frame      = anim->numFrames() ? position % anim->numFrames() : 0;
    anim->previewed_frame_time = 0.0f;
  }
}
