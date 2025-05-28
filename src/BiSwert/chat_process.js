let filesArray = [];

$(document).ready(function () {
    /* textarea handling logic */
    $('textarea').on('input', function () {
        $(this).css('height', 'auto');
        $(this).css('height', this.scrollHeight + 'px'); 
    });

    /* upload logic */
    $('#file-upload').on('change', function (e) {
        const newFiles = Array.from(e.target.files);
        const filteredFiles = newFiles.filter(file => !isDuplicateFile(file));

        filesArray = filesArray.concat(filteredFiles);

        renderFileList();
        removeHandler();
    });

    /* send logic */
    $('#send-button').on('click', function (e) {
        e.preventDefault();
        $('#send-button').prop('disabled', true);

        const $input = $('#chat-input');
        const text = $.trim($input.val());
        const files = filesArray;

        if (!validationTest(text, files)) {
            $('#send-button').prop('disabled', false);
            return;
        }

        UsrMsgHandler(text, files);
        LLMMsgHandler(text, files);


        $('html, body').animate({ scrollTop: $(document).height() }, 1000, 'swing');
        setInputHeight();
        $input.val('');
    });
});

/* ====================================== file-upload function ====================================== */

function isDuplicateFile(file) {
    const exists = filesArray.some(f => f.name === file.name);
    if (exists) {
        alert(`이미 추가된 파일입니다: ${file.name}`);
    }
    return exists;
}

function renderFileList() {
    $('#file-list').empty();
    filesArray.forEach(file => {
        $('#file-list').append(`
            <li>
                <i class="fa-solid fa-paperclip clip-icon"></i>
                ${file.name}
                <i class="fa-solid fa-xmark remove-icon" data-name="${file.name}"></i>
            </li>
        `);
    });
    $('#file-upload').val('');
    
    $('#file-floating-panel').show();
}

function removeHandler() {
    $('.remove-icon').off('click').on('click', function() {
        if (confirm('제거하시겠습니까?')) {
            const fileName = $(this).data('name');
            filesArray = filesArray.filter(f => f.name !== fileName);
            $(this).closest('li').remove();
        }
    });
}


/* ====================================== send function ====================================== */

function setInputHeight() {
    $('#chat-input').css('height', 'auto');
    $('#chat-input').css('height', $('#chat-input').scrollHeight + 'px');
}

function validationTest(text, files) {
    if (text.length === 0 && files.length === 0) return false;
    return true;
}

/* ---------------- UserMsg ---------------- */
function UsrMsgHandler(text, files) {
    const time = getTime();

    let fileNames = '';
    if (files.length) {
        fileNames = '<ul class="file-name-list">';
        for (let i = 0; i < files.length; i++) {
            fileNames += `<li> 
                <i class="fa-solid fa-paperclip clip-icon"></i> 
                ${escapeHtml(files[i].name)}
                </li>`;
        }
        fileNames += '</ul><hr>';
    }

    const usrMsg = $(`
    <div class="chat-message user-message">
      <div class="message-meta">나 ・ ${time}</div>
      <hr>
      ${fileNames}
      <pre class="message-content">${escapeHtml(text)}</pre>
    </div>
  `);

    $('#chat-container').append(usrMsg);

    setTimeout(() => {
        usrMsg.addClass('show');
    }, 10);
}

/* ---------------- LLMMsg ---------------- */
function LLMMsgHandler(text, files) {
    const llmMsg = makeLLMMsg();

    const resultHtml = sendToLLM(text, files);

    setTimeout(() => {
        console.log("test loading");
        typingMsg(llmMsg, resultHtml);
    }, 1000);
}

function makeLLMMsg() {
    const time = getTime();

    const llmMsg = $(`
    <div class="chat-message bot-message">
        <div class="message-meta">BiSwert ・ ${time}</div>
        <hr>
        <pre class="message-content"><div class="loading-spinner"></div></pre>
    </div>
`);

    $('#chat-container').append(llmMsg);

    setTimeout(() => {
        llmMsg.addClass('show');
    }, 500);

    return llmMsg;
}

function sendToLLM(text, files) {
    /* 구현 방향
    * text를 독립적인 new_file.c 로 만들어서 (file 이름은 files와 겹치지 않게 Random으로 만들어서 처리하든..)
    * llm에게 fileArray (new_file.c 가 포함된) 보내기
    * llm으로부터 반환 받은 결과를 html로 변환 후 html 반환
    */    

    /* 구현 위치 */

    let allFiles = files;

    // 1. 랜덤 파일명 생성 (files 이름과 중복 안 되게)

    if(text !== "") {
        const newFileName = generateRandomFileName(files);
        
        const blob = new Blob([text], { type: 'text/plain' });
        const newFile = new File([blob], newFileName, {
            type: 'text/plain',
            lastModified: Date.now()
        });
        allFiles = files.concat(newFile);
    }
    
    /* 구현 위치 */

    let fileNames = '';
    if (files.length) {
        for (let i = 0; i < files.length; i++) {
            fileNames += `${escapeHtml(files[i].name)}\n`
        }
        
        filesArray = [];
        renderFileList();
        $('#file-floating-panel').hide();
    }

    const html = fileNames + text; // 반환 값

    return html;
}

function generateRandomFileName(existingFiles) {
    let name;
    do {
        name = 'new_file_' + Math.floor(Math.random() * 10000) + '.s';
    } while (existingFiles.some(f => f.name === name));
    return name;
}

function typingMsg(llmMsg, html) { // files 임의 사용
    llmMsg.find('.message-content').empty()

    let index = 0;

    function type() {
        if (index < html.length) {
            llmMsg.find('.message-content').append(html.charAt(index));
            index++;
            setTimeout(type, 10);
        } else {
            $('#send-button').prop('disabled', false);
        }
    }

    type();
}

function getTime() {
    const now = new Date();
    const time = now.getHours().toString().padStart(2, '0') + ':' +
        now.getMinutes().toString().padStart(2, '0');

    return time;
}

function escapeHtml(text) {
    return $('<div>').text(text).html();
}
