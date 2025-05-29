// API 설정
const API_CONFIG = {
    BASE_URL: 'http://168.188.126.228:8000',
    ANALYZE_ENDPOINT: '/api/analyze/'
};

// 전역 변수
let filesArray = [];
let inputCounter = 1;

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
        files.forEach(file => {
            fileNames += `<li> 
                <i class="fa-solid fa-paperclip clip-icon"></i> 
                ${escapeHtml(file.name)}
                </li>`;
        });
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
    
    // 모든 파일에 대한 분석 요청을 Promise 배열로 생성
    const requests = [];
    
    // 텍스트 입력이 있는 경우
    if (text.trim() !== "") {
        requests.push(sendToLLM(text, []));
    }
    
    // 파일들에 대해 각각 요청
    files.forEach(file => {
        requests.push(sendToLLM("", [file]));
    });

    // 모든 요청의 결과를 처리
    Promise.all(requests)
        .then(results => {
            // 파일 목록 초기화
            filesArray = [];
            renderFileList();
            $('#file-floating-panel').hide();

            // 결과들을 하나의 문자열로 합침
            const combinedResult = results.join('\n\n');
            setTimeout(() => {
                typingMsg(llmMsg, combinedResult);
            }, 1000);
        })
        .catch(error => {
            console.error('Error:', error);
            typingMsg(llmMsg, '오류가 발생했습니다: ' + error.message);
        });
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

function getLabelDescription(label) {
    const descriptions = {
        "switch_origin": "- switch가 있는 원본 코드입니다.",
        "switch_flat": "- switch가 있는 평탄화 난독화 코드입니다.",
        "switch_opaque": "- switch가 있는 불투명 술어 난독화 코드입니다.",
        "switch_vir": "- switch가 있는 가상화 난독화 코드입니다.",
        "non_switch_origin": "- switch가 없는 원본 코드입니다.",
        "non_switch_flat": "- switch가 없는 평탄화 난독화 코드입니다.",
        "non_switch_opaque": "- switch가 없는 불투명 술어 난독화 코드입니다.",
        "non_switch_vir": "- switch가 없는 가상화 난독화 코드입니다."
    };
    return descriptions[label] || "알 수 없는 유형의 코드입니다.";
}

function formatApiResponse(data, fileInfo) {
    if (!data.success) {
        return "분석 중 오류가 발생했습니다.";
    }
    
    // 파일명 표시 (텍스트 입력인 경우 제외)
    const fileName = fileInfo.fileName ? `[${fileInfo.fileName}]\n` : "";
    return `${fileName}${getLabelDescription(data.predictedLabel)}`;
}

function extractFileNameFromText(text) {
    // 첫 줄에서 파일 이름 추출 시도
    const firstLine = text.trim().split('\n')[0];
    const match = firstLine.match(/\.file\s*"([^"]+)"/);
    if (match) {
        return match[1];
    }
    // 파일 이름이 없는 경우 순차적인 번호 사용
    return `input_${inputCounter++}.s`;
}

function sendToLLM(text, files) {
    return new Promise((resolve, reject) => {
        const formData = new FormData();
        let fileInfo = { fileName: null };
        
        // 텍스트가 있는 경우 .s 파일로 추가
        if(text !== "") {
            const newFileName = extractFileNameFromText(text);
            const blob = new Blob([text], { type: 'text/plain' });
            const newFile = new File([blob], newFileName, {
                type: 'text/plain',
                lastModified: Date.now()
            });
            formData.append('file', newFile);
            fileInfo.fileName = newFileName;
        }

        // 파일 추가
        if (files.length > 0) {
            fileInfo.fileName = files[0].name;
            formData.append('file', files[0]);
        }

        // API 호출
        fetch(API_CONFIG.BASE_URL + API_CONFIG.ANALYZE_ENDPOINT, {
            method: 'POST',
            body: formData,
            // CORS 관련 설정 추가
            mode: 'cors',
            credentials: 'same-origin'
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // API 응답을 사용자 친화적인 메시지로 변환
            resolve(formatApiResponse(data, fileInfo));
        })
        .catch(error => {
            console.error('Error:', error);
            let errorMessage = '오류가 발생했습니다: ';
            
            // 에러 메시지 상세화
            if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
                errorMessage += '서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.';
            } else if (error.message.includes('HTTP error')) {
                errorMessage += '서버 응답 오류가 발생했습니다. 잠시 후 다시 시도해주세요.';
            } else {
                errorMessage += error.message;
            }
            
            resolve(errorMessage);
        });
    });
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
