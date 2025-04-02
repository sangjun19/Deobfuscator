#include "QtView.h"
#include "TodoModel.h"

#include <QHeaderView>
#include <QMenu>
#include <QTableWidgetItem>
#include <QVBoxLayout>
#include <qcombobox.h>

QtView::QtView(QWidget* parent) : QMainWindow(parent)
{
	m_CentralWidget = new QWidget(this);
	setCentralWidget(m_CentralWidget);

	m_TitleLine   = new QLineEdit(this);
	m_DescText    = new QTextEdit(this);
	m_StatusCombo = new QComboBox(this);
	m_StatusCombo->addItem("ToDo");
	m_StatusCombo->addItem("InProgress");
	m_StatusCombo->addItem("Done");

	m_AddButton = new QPushButton("Add Task", this);
	m_Table     = new QTableWidget(this);
	m_Table->setColumnCount(4);
	m_Table->setHorizontalHeaderLabels(QStringList() << "ID" << "Title" << "Description" << "Status");
	m_Table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
	m_Table->setEditTriggers(QAbstractItemView::DoubleClicked | QAbstractItemView::SelectedClicked);
	m_Table->setContextMenuPolicy(Qt::CustomContextMenu);

	QVBoxLayout* layout{new QVBoxLayout(m_CentralWidget)};
	layout->addWidget(m_TitleLine);
	layout->addWidget(m_DescText);
	layout->addWidget(m_StatusCombo);
	layout->addWidget(m_AddButton);
	layout->addWidget(m_Table);

	connect(m_AddButton, &QPushButton::clicked, this, &QtView::OnAddButtonClicked);
	connect(m_Table, &QTableWidget::cellChanged, this, &QtView::OnCellChanged);
	connect(m_Table, &QTableWidget::customContextMenuRequested, this, &QtView::OnCustomContextMenuRequested);
}

void QtView::UpdateTasks(const std::unordered_map<TodoModel::TaskID, Task>& tasks)
{
	QSignalBlocker blocker{m_Table};        // Prevent unwanted signals during update
	m_Table->clearContents();
	m_Table->setRowCount(static_cast<int>(tasks.size()));

	int row{0};
	for (const auto& [id, task] : tasks)
	{
		// Column 0: ID (non-editable)
		QTableWidgetItem* idItem{new QTableWidgetItem(QString::number(id))};
		idItem->setFlags(idItem->flags() & ~Qt::ItemIsEditable);
		m_Table->setItem(row, 0, idItem);

		// Column 1: Title (editable)
		QTableWidgetItem* titleItem{new QTableWidgetItem(QString::fromStdString(task.Title))};
		m_Table->setItem(row, 1, titleItem);

		// Column 2: Description (editable)
		QTableWidgetItem* descItem{new QTableWidgetItem(QString::fromStdString(task.Description))};
		m_Table->setItem(row, 2, descItem);

		// Column 3: Status (with QComboBox)
		QComboBox* combo{CreateStatusComboBox(task.Status, id, row)};
		m_Table->setCellWidget(row, 3, combo);

		row++;
	}
}

QComboBox* QtView::CreateStatusComboBox(TaskStatus currentStatus, TodoModel::TaskID id, int row)
{
	QComboBox* combo{new QComboBox()};
	combo->addItem("ToDo");
	combo->addItem("InProgress");
	combo->addItem("Done");

	int currentIndex{0};
	switch (currentStatus)
	{
		case TaskStatus::ToDo:
			currentIndex = 0;
			break;
		case TaskStatus::InProgress:
			currentIndex = 1;
			break;
		case TaskStatus::Done:
			currentIndex = 2;
			break;
	}
	combo->setCurrentIndex(currentIndex);
	combo->setProperty("taskId", QVariant::fromValue<qulonglong>(id));
	combo->setProperty("row", row);

	connect(combo, QOverload<int>::of(&QComboBox::currentIndexChanged), [this, combo](int index) {
		bool              ok;
		TodoModel::TaskID taskId = combo->property("taskId").toULongLong(&ok);
		if (!ok)
			return;
		int row = combo->property("row").toInt();
		// Retrieve updated Title and Description from the table
		QTableWidgetItem* titleItem = m_Table->item(row, 1);
		QTableWidgetItem* descItem  = m_Table->item(row, 2);
		QString           title     = titleItem ? titleItem->text() : "";
		QString           desc      = descItem ? descItem->text() : "";
		TaskStatus        newStatus = TaskStatus::ToDo;
		if (index == 1)
			newStatus = TaskStatus::InProgress;
		else if (index == 2)
			newStatus = TaskStatus::Done;
		emit UpdateTaskRequested(taskId, title, desc, newStatus);
	});
	return combo;
}

void QtView::OnAddButtonClicked()
{
	QString    title{m_TitleLine->text().trimmed()};
	QString    desc{m_DescText->toPlainText().trimmed()};
	int        statusIndex{m_StatusCombo->currentIndex()};
	TaskStatus status{TaskStatus::ToDo};

	if (statusIndex == 1)
		status = TaskStatus::InProgress;
	else if (statusIndex == 2)
		status = TaskStatus::Done;

	if (!title.isEmpty())
	{
		emit AddTaskRequested(title, desc, status);
		m_TitleLine->clear();
		m_DescText->clear();
		m_StatusCombo->setCurrentIndex(0);
	}
}

void QtView::OnCellChanged(int row, int column)
{
	// We do not update for ID or Status columns (Status is handled by the combo box)
	if (column == 0 || column == 3)
		return;

	QTableWidgetItem* idItem{m_Table->item(row, 0)};
	if (!idItem)
		return;

	bool              ok;
	TodoModel::TaskID taskId = idItem->text().toULongLong(&ok);
	if (!ok)
		return;

	// Retrieve Title and Description from the current row
	QTableWidgetItem* titleItem{m_Table->item(row, 1)};
	QTableWidgetItem* descItem{m_Table->item(row, 2)};
	QString           title{titleItem ? titleItem->text() : ""};
	QString           desc{descItem ? descItem->text() : ""};

	// For status, try to get the QComboBox from column 3
	TaskStatus status = TaskStatus::ToDo;
	QWidget*   widget = m_Table->cellWidget(row, 3);
	if (auto combo = qobject_cast<QComboBox*>(widget))
	{
		int idx = combo->currentIndex();
		if (idx == 1)
			status = TaskStatus::InProgress;
		else if (idx == 2)
			status = TaskStatus::Done;
	}

	emit UpdateTaskRequested(taskId, title, desc, status);
}

void QtView::OnCustomContextMenuRequested(const QPoint& pos)
{
	QModelIndex index{m_Table->indexAt(pos)};
	if (!index.isValid())
		return;
	int               row{index.row()};
	QTableWidgetItem* idItem{m_Table->item(row, 0)};
	if (!idItem)
		return;
	bool              ok;
	TodoModel::TaskID taskId{idItem->text().toULongLong(&ok)};
	if (!ok)
		return;
	QMenu    contextMenu;
	QAction* removeAction{contextMenu.addAction("Remove Task")};
	QAction* selectedAction{contextMenu.exec(m_Table->viewport()->mapToGlobal(pos))};
	if (selectedAction == removeAction)
		emit RemoveTaskRequested(taskId);
}
