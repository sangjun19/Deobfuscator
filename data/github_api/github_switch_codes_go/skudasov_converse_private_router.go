// Repository: skudasov/converse
// File: bot/private_router.go

package bot

import (
	"bytes"
	"fmt"
	"github.com/f4hrenh9it/converse/config"
	"github.com/f4hrenh9it/converse/db"
	"github.com/f4hrenh9it/converse/log"
	"github.com/go-telegram-bot-api/telegram-bot-api"
	"strings"
	"time"
)

const (
	StartCommand                    = "start"
	ConversationActiveSelectorMenu  = "active"
	ConversationHistorySelectorMenu = "history"
	ConversationCurrentMenu         = "current"
	ConversationsListNonResolved    = "list"
	MyChatIdCommand                 = "mychatid"
	//Agents only
	ConversationSearch = "search"
)

func (m *Bot) SendHistory(chatId int64, hist []db.TypedMsg) error {
	for _, h := range hist {
		var buf bytes.Buffer
		//header
		buf.WriteString(fmt.Sprintf("*%s*\n", h.UserType))
		switch h.MsgType {
		case "T":
			buf.WriteString(fmt.Sprintf("`%s`\n", h.Msg))

			msg := tgbotapi.NewMessage(chatId, buf.String())
			msg.ParseMode = "markdown"
			m.ResponseChan <- msg
		case "P":
			// document or photo had only caption for text, so print header anyway
			msg := tgbotapi.NewMessage(chatId, buf.String())
			msg.ParseMode = "markdown"
			m.ResponseChan <- msg
			log.L.Debugf("sending photo")
			ps := tgbotapi.NewPhotoShare(chatId, h.AttachId)
			ps.Caption = h.Caption
			m.ResponseChan <- ps
		case "D":
			msg := tgbotapi.NewMessage(chatId, buf.String())
			msg.ParseMode = "markdown"
			m.ResponseChan <- msg
			fs := tgbotapi.NewDocumentShare(chatId, h.AttachId)
			fs.Caption = h.Caption
			m.ResponseChan <- fs
		}
	}
	return nil
}

func (m *Bot) SendConversationList(chatId int64, convList []*db.ListConvInfo) error {
	var buf bytes.Buffer
	var msg tgbotapi.MessageConfig
	if len(convList) == 0 {
		msg = conversationsListEmpty(chatId)
	} else {
		for _, p := range convList {
			if p.TotalTime > time.Hour*time.Duration(config.B.DefaultConversationSla) {
				buf.WriteString(fmt.Sprintf("*Id*: `%d`\n*Description*: `%s`\n*Total time*: `%s`\n\n",
					p.Id, p.Description, "12h sla broken"))
			} else {
				buf.WriteString(fmt.Sprintf("*Id*: `%d`\n*Description*: `%s`\n*Total time*: `%s`\n\n",
					p.Id, p.Description, p.TotalTime))
			}
		}
		msg = tgbotapi.NewMessage(chatId, buf.String())
	}
	msg.ParseMode = "markdown"
	m.ResponseChan <- msg
	return nil
}

func (m *Bot) SendSearchResults(chatid int64, searchResults []*db.SearchResult) error {
	for _, sr := range searchResults {
		msg := searchResult(chatid, sr)
		m.ResponseChan <- msg
	}
	return nil
}

func argsToList(args string) []string {
	return strings.Split(args, " ")
}

func (m *Bot) HandleCmd(update tgbotapi.Update) error {
	chatId := update.Message.Chat.ID

	cmd := update.Message.Command()
	argString := update.Message.CommandArguments()

	isAgent := db.IsAgent(chatId)

	log.L.Debugf("entities: %s", update.Message.Entities)

	switch cmd {
	case StartCommand:
		var msg tgbotapi.MessageConfig
		if isAgent {
			msg = startMsgAgent(chatId)
		} else {
			msg = startMsg(chatId)
		}
		m.ResponseChan <- msg

		if !db.UserExists(update) {
			if err := m.NewConversation(chatId, isAgent); err != nil {
				return fmt.Errorf(NewConvErr, err)
			}
		}
		err := db.RegisterOrUpdateUser(update)
		if err != nil {
			return fmt.Errorf("failed to register or update user data: %s", err)
		}
	case ConversationSearch:
		var msg tgbotapi.MessageConfig
		if !isAgent {
			msg = searchNotAllowed(chatId)
			m.ResponseChan <- msg
			return nil
		}
		if len(argString) == 0 {
			msg = nothingToSearch(chatId)
			m.ResponseChan <- msg
			return nil
		}
		res, err := db.Search(argString)
		if err != nil {
			return fmt.Errorf(SearchErr, err)
		}

		if err := m.SendSearchResults(chatId, res); err != nil {
			return fmt.Errorf(SendSearchResultsErr, err)
		}
	case MyChatIdCommand:
		msg := myChatId(chatId)
		m.ResponseChan <- msg
	case ConversationActiveSelectorMenu:
		convs, err := db.GetActiveConversations(isAgent, chatId)
		if err != nil {
			return fmt.Errorf(TgApiErr, err)
		}
		msg := ConversationSelectorKb(chatId, convs, true)

		m.ResponseChan <- msg
	case ConversationHistorySelectorMenu:
		convs, err := db.GetHistoryConversations(isAgent, chatId)
		if err != nil {
			return err
		}
		msg := ConversationSelectorKb(chatId, convs, false)

		m.ResponseChan <- msg
	case ConversationCurrentMenu:
		if _, ok := CS.CurrentConversation[chatId]; !ok {
			msg := notInAnyConversation(chatId)
			m.ResponseChan <- msg
			return nil
		}
		msg := ConversationCurrentKb(chatId, CS.CurrentConversation[chatId])
		m.ResponseChan <- msg
	default:
		msg := unrecognizedCommand(chatId)
		m.ResponseChan <- msg
	}
	CS.Debug()
	return nil
}

func extractAttach(update tgbotapi.Update) (caption string, attachId string, attachType string, err error) {
	caption = update.Message.Caption
	switch {
	case update.Message.Photo != nil:
		photos := *update.Message.Photo
		//extract biggest resolution photo
		attachId = photos[len(photos)-1].FileID
		attachType = "P"
		log.L.Debugf("photo attachId: %s", attachId)
	case update.Message.Document != nil:
		attachId = update.Message.Document.FileID
		attachType = "D"
		log.L.Debugf("document attachId: %s", attachId)
	default:
		attachType = "T"
		log.L.Debugf("text msg")
	}
	return
}

func (m *Bot) AlertVisited(chatId int64, convId int) error {
	name := db.NameByChatId(chatId)
	msg := visitAlert(CS.SupportChat, convId, name)
	m.ResponseChan <- msg
	return nil
}

func (m *Bot) AlertNewMsg(toChat int64, fromId int64, convId int, isAgent bool) error {
	var msg tgbotapi.MessageConfig
	desc := CS.Conversations[convId].Description
	if isAgent {
		name := db.NameByChatId(fromId)
		msg = newAnswerAlert(toChat, convId, desc, name)
	} else {
		name := db.NameByChatId(fromId)
		msg = newQuestionAlert(toChat, convId, desc, name)
	}
	msg.ParseMode = "markdown"
	m.ResponseChan <- msg
	return nil
}

func (m *Bot) HandleMsg(update tgbotapi.Update) error {
	fromId := update.Message.Chat.ID
	text := update.Message.Text
	isAgent := db.IsAgent(fromId)

	log.L.Debugf("msg from chatid: %d", fromId)
	curConvId, ok := CS.CurrentConversation[fromId]
	if !ok {
		msg := notInAnyConversation(fromId)
		m.ResponseChan <- msg
		return nil
	}

	curConv := CS.Conversations[curConvId]
	caption, attachId, attachType, err := extractAttach(update)
	if err != nil {
		return err
	}

	if !db.ConvActive(curConvId) {
		msg := conversationClosed(fromId, curConvId)
		m.ResponseChan <- msg
		return nil
	}
	if curConv.Description == "" {
		curConv.Description = text
	}
	if err := m.AlertNewMsg(CS.SupportChat, fromId, curConvId, isAgent); err != nil {
		return fmt.Errorf(AlertNewMsgErr, err)
	}

	if err := db.WitnessMsg(curConvId, fromId, text, attachId, attachType, caption); err != nil {
		return err
	}
	if curConv.UserId != 0 && curConv.AgentId != 0 {
		if err := curConv.Deliver(B, update, isAgent); err != nil {
			return fmt.Errorf(MsgDeliveryErr, err)
		}
	} else if curConv.UserId == 0 {
		lastSeenUser := CS.Conversations[curConvId].LastSeenUserId
		if err := m.AlertNewMsg(lastSeenUser, fromId, curConvId, isAgent); err != nil {
			return fmt.Errorf(AlertNewMsgErr, err)
		}
	}
	return nil
}

func (m *Bot) PrivateRoute(update tgbotapi.Update) error {
	if update.Message.IsCommand() {
		if err := m.HandleCmd(update); err != nil {
			log.L.Error(err)
		}
	} else {
		if err := m.HandleMsg(update); err != nil {
			log.L.Error(err)
		}
	}
	return nil
}
