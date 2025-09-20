// static/js/game.js

const moveQualityMap = {
    blunder: {
        icon: '/static/img/symbols/blunder.png',
        text: 'Blunder'
    },
    mistake: {
        icon: '/static/img/symbols/mistake.png',
        text: 'Mistake'
    },
    inaccuracy: {
        icon: '/static/img/symbols/inaccuracy.png',
        text: 'Inaccuracy'
    },
    good: {
        icon: '/static/img/symbols/good.png',
        text: 'Good Move'
    },
    excellent: {
        icon: '/static/img/symbols/excellent.png',
        text: 'Excellent Move'
    },
    best: {
        icon: '/static/img/symbols/best.png',
        text: 'Best Move'
    }
};

$(document).ready(function() {
    let socket = null;
    let analysisRequestCounter = 0;
    
    let isAiThinking = false;
    let lastAiCommentaryPly = -1;

    let currentAudio = null;
    let isAudioEnabled = false; 

    // --- NEW: Caching objects ---
    let moveQualityCache = {};
    let analysisCache = {};
    let commentaryCache = {};

    let stockfishSettings = {
        depth: 18,
        threads: 4
    };
    
    const $arrowContainer = $('#arrow-container');
    
    function clearArrows() {
        $arrowContainer.empty();
    }

    function drawArrow(from, to) {
        const boardEl = document.getElementById('board');
        const arrowContainerEl = document.getElementById('arrow-container');
        const fromEl = boardEl.querySelector(`.square-${from}`);
        const toEl = boardEl.querySelector(`.square-${to}`);

        if (!boardEl || !arrowContainerEl || !fromEl || !toEl) return;

        const boardRect = boardEl.getBoundingClientRect();
        if (boardRect.width === 0) return;

        const arrowContainerRect = arrowContainerEl.getBoundingClientRect();
        const fromRect = fromEl.getBoundingClientRect();
        const toRect = toEl.getBoundingClientRect();

        const fromCenter = {
            x: (fromRect.left - arrowContainerRect.left) + (fromRect.width / 2),
            y: (fromRect.top - arrowContainerRect.top) + (fromRect.height / 2),
        };
        const toCenter = {
            x: (toRect.left - arrowContainerRect.left) + (toRect.width / 2),
            y: (toRect.top - arrowContainerRect.top) + (toRect.height / 2),
        };

        const dx = toCenter.x - fromCenter.x;
        const dy = toCenter.y - fromCenter.y;
        const length = Math.sqrt(dx * dx + dy * dy);
        const angleRad = Math.atan2(dy, dx);
        
        const squareSize = boardRect.width / 8;
        const headWidth = squareSize * 0.5;
        const headHeight = squareSize * 0.7; 
        const shaftHeight = squareSize * 0.30; 
        
        const $wrapper = $('<div>').addClass('arrow-wrapper').css({
            left: fromCenter.x,
            top: fromCenter.y,
            width: length,
            height: headHeight,
            transform: `rotate(${angleRad}rad)`,
        });

        const shaftLength = length - (headWidth * 0.8);

        const $shaft = $('<div>').addClass('arrow-shaft').css({
             height: shaftHeight,
             width: shaftLength > 0 ? shaftLength : 0,
        });

        const $head = $('<div>').addClass('arrow-head').css({
            width: headWidth,
            height: headHeight,
        });

        $wrapper.append($shaft, $head);
        $arrowContainer.append($wrapper);
    }

    function setAnalysisLoadingState(isLoading) {
        if (isLoading) {
            $('#analysis-loader').removeClass('hidden');
            $('#evaluation-score').text('...');
        } else {
            $('#analysis-loader').addClass('hidden');
        }
    }

    function setAiCommentaryLoading(isLoading) {
        const $commentaryEl = $('#ai-commentary');
        if (isLoading) {
            isAiThinking = true;
            $commentaryEl.html('<span class="italic text-text-secondary">AI is thinking...</span>');
        } else {
            isAiThinking = false;
        }
    }
    
    function displayMoveQualityIcon(classification, move) {
        const $indicator = $('#move-quality-indicator');
        const $icon = $('#move-quality-icon');
        const quality = moveQualityMap[classification];

        if (!quality || !move) {
            $indicator.addClass('opacity-0');
            return;
        }

        const toSquare = move.to;
        const fileIndex = toSquare.charCodeAt(0) - 'a'.charCodeAt(0);
        const rankIndex = 8 - parseInt(toSquare.charAt(1));

        const left = (fileIndex * 12.5) + '%';
        const top = (rankIndex * 12.5) + '%';
        
        $indicator.css({ left: left, top: top });
        $icon.attr('src', quality.icon);
        $indicator.attr('title', quality.text);
        $indicator.removeClass('opacity-0');
    }
    
    function hideMoveQualityIcon() {
        $('#move-quality-indicator').addClass('opacity-0');
    }

    function displayCommentary(commentary) {
        let formattedCommentary = (commentary || 'Could not retrieve commentary.')
            .replace(/\*\*(.*?)\*\*/g, '<strong class="text-white font-semibold">$1</strong>')
            .replace(/\*(.*?)\*/g, '<em class="italic">$1</em>')
            .replace(/\n/g, '<br>');
        $('#ai-commentary').html(formattedCommentary);
    }

    if (stockfishEnabled) {
        socket = io();

        socket.on('analysis_result', function(data) {
            if (data.requestId !== analysisRequestCounter) { return; }
            updateAnalysisUI(data);

            const isFinalDepth = data.depth === stockfishSettings.depth;

            // Only cache and request commentary on the final depth result
            if (isFinalDepth) {
                setAnalysisLoadingState(false);
                analysisCache[currentMoveIndex] = data; // Store in cache

                if (commentaryCache[currentMoveIndex]) {
                    displayCommentary(commentaryCache[currentMoveIndex]);
                } else if (!isAiThinking && currentMoveIndex !== lastAiCommentaryPly) {
                    lastAiCommentaryPly = currentMoveIndex;
                    setAiCommentaryLoading(true);

                    let pgn = '';
                    for (let i = 0; i <= currentMoveIndex; i++) {
                        if (i % 2 === 0) { pgn += `${Math.floor(i / 2) + 1}. `; }
                        pgn += `${history[i].san} `;
                    }

                    const commentaryPayload = {
                        fen: fens[currentMoveIndex + 1], pgn: pgn.trim(), ply: currentMoveIndex + 1,
                        humanMove: history[currentMoveIndex].san,
                        engineBestMove: data.moves.length > 0 ? data.moves[0].san : 'N/A',
                        evaluation: data.eval, topLines: data.moves,
                        audio_enabled: isAudioEnabled
                    };
                    socket.emit('get_ai_commentary', commentaryPayload);
                }
            }
        });

        socket.on('move_quality_result', function(data) {
            if (data.classification) {
                moveQualityCache[currentMoveIndex] = data.classification; // Store in cache
                displayMoveQualityIcon(data.classification, history[currentMoveIndex]);
            }
        });

        socket.on('analysis_error', function(data) {
            if (data.requestId && data.requestId !== analysisRequestCounter) { return; }
            setAnalysisLoadingState(false);
            clearArrows();
            console.error('Analysis Error:', data.message);
            $('#top-lines-display').html(`<p class="text-xs text-red-400 font-mono">${data.message}</p>`);
        });

        socket.on('ai_commentary_text_result', function(data) {
            setAiCommentaryLoading(false);
            commentaryCache[currentMoveIndex] = data.commentary; // Store in cache
            displayCommentary(data.commentary);
        });
        
        socket.on('ai_commentary_audio_result', function(data) {
            if (!isAudioEnabled) { return; } 

            if (currentAudio) {
                currentAudio.pause();
                currentAudio.currentTime = 0;
            }

            if (data.audio_data) {
                const audioSrc = 'data:audio/wav;base64,' + data.audio_data;
                currentAudio = new Audio(audioSrc);
                currentAudio.play().catch(e => console.error("Audio playback failed:", e));
            }
        });
        
        socket.on('ai_commentary_error', function(data) {
            setAiCommentaryLoading(false);
            $('#ai-commentary').html(`<span class="text-xs text-red-400 font-mono">${data.message}</span>`);
        });

        socket.on('full_analysis_complete', function(data) {
            $('#accuracy-panel').hide();
            $('#header-accuracy-loading').hide();
            $('#header-accuracy-results').removeClass('hidden').addClass('flex');
            $('#accuracy-white').text(`${data.white}%`);
            $('#accuracy-black').text(`${data.black}%`);
        });
    }

    function requestAnalysis(fen, moveIndex) {
        if (!socket || !stockfishEnabled) return;

        // --- NEW: Check cache before making requests ---
        if (analysisCache[moveIndex]) {
            setAnalysisLoadingState(false);
            updateAnalysisUI(analysisCache[moveIndex]);
            if (commentaryCache[moveIndex]) {
                displayCommentary(commentaryCache[moveIndex]);
            }
            if (moveQualityCache[moveIndex]) {
                displayMoveQualityIcon(moveQualityCache[moveIndex], history[moveIndex]);
            }
            return; // Exit if we are using cached data
        }

        // --- Original logic if not cached ---
        if (moveIndex > -1) {
            $('#ai-commentary').html('<span class="italic text-text-secondary">Waiting for engine analysis...</span>');
        } else {
             $('#ai-commentary').html('<span class="italic text-text-secondary">Navigate moves to see AI commentary.</span>');
        }

        const requestId = ++analysisRequestCounter;
        setAnalysisLoadingState(true);
        clearArrows();
        socket.emit('analyze_position', {
            fen: fen,
            settings: stockfishSettings,
            requestId: requestId
        });

        if (moveIndex >= 0) {
            socket.emit('get_move_quality', {
                fen_before: fens[moveIndex],
                fen_after: fens[moveIndex + 1]
            });
        }
    }

    function updateAnalysisUI(data) {
        const evalScoreEl = $('#evaluation-score');
        const evalBarWhite = $('#evaluation-bar-white');
        if (data.eval.type === 'cp') {
            const score = (data.eval.value / 100.0).toFixed(2);
            evalScoreEl.text(score > 0 ? `+${score}` : score);
            const clampedScore = Math.max(-800, Math.min(800, data.eval.value));
            const percentage = 50 + (clampedScore / 800) * 50;
            evalBarWhite.css('height', `${percentage}%`);
        } else if (data.eval.type === 'mate') {
            const mateIn = data.eval.value;
            evalScoreEl.text(mateIn > 0 ? `M${mateIn}` : `M-${Math.abs(mateIn)}`);
            evalBarWhite.css('height', mateIn > 0 ? '100%' : '0%');
        }
        const topLinesDisplay = $('#top-lines-display');
        if (data.moves && data.moves.length > 0) {
            clearArrows();
            const topMove = data.moves[0];
            if (topMove && topMove.uci) {
                const from = topMove.uci.substring(0, 2);
                const to = topMove.uci.substring(2, 4);
                drawArrow(from, to);
            }
            let linesHtml = `<div class="w-full font-mono text-text-secondary"><div class="text-xs text-right pb-1 border-b border-border-primary/50 mb-1">Depth: ${data.depth}</div>`;
            data.moves.forEach((move, index) => {
                let scoreText = '';
                if (move.mate !== null) {
                    scoreText = `M${move.mate}`;
                } else if (move.cp !== null) {
                    const cp_score = (move.cp / 100.0).toFixed(2);
                    scoreText = cp_score > 0 ? `+${cp_score}` : cp_score;
                }
                linesHtml += `<div class="flex justify-between items-center py-0.5"><span class="text-text-primary">${index + 1}. ${move.san}</span><span>${scoreText}</span></div>`;
            });
            linesHtml += '</div>';
            topLinesDisplay.html(linesHtml);
        } else {
            clearArrows();
            topLinesDisplay.html('<p class="text-xs text-text-secondary">No moves found.</p>');
        }
    }

    function formatTime(totalSeconds) {
        if (typeof totalSeconds !== 'number' || isNaN(totalSeconds)) { return "00:00"; }
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = Math.floor(totalSeconds % 60);
        return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }

    const game = new Chess(startFen || undefined);
    const history = [];
    if (movesData) {
        movesData.forEach(move => {
            const moveResult = game.move(move.san);
            if (moveResult) { history.push(moveResult); }
        });
    } else {
        console.error("movesData is not available. Check for server-side template errors.");
    }
    game.reset();
    const fens = [startFen || new Chess().fen()];
    history.forEach(move => { game.move(move.san); fens.push(game.fen()); });
    game.reset();
    if (socket && stockfishEnabled) {
        socket.emit('request_full_analysis', { fens: fens });
    }
    const whiteClockHistory = [initialTimeSeconds];
    const blackClockHistory = [initialTimeSeconds];
    let lastWhiteClock = initialTimeSeconds;
    let lastBlackClock = initialTimeSeconds;
    if (movesData) {
        movesData.forEach(move => {
            if (move.ply % 2 !== 0) { lastWhiteClock = move.clock; } 
            else { lastBlackClock = move.clock; }
            whiteClockHistory.push(lastWhiteClock);
            blackClockHistory.push(lastBlackClock);
        });
    }
    let currentMoveIndex = -1;
    let board = null;
    const moveListContainer = $('#move-list-container');
    let moveListHtml = '<div class="space-y-1">';
    for (let i = 0; i < history.length; i += 2) {
        let fullMoveNumber = i / 2 + 1;
        moveListHtml += `<div class="grid grid-cols-[30px_1fr_1fr] gap-x-2 items-center text-base"><span class="text-text-secondary text-right">${fullMoveNumber}.</span><span class="font-medium move-item p-1.5 rounded-md cursor-pointer transition-colors duration-150 hover:bg-tertiary" data-move-index="${i}">${history[i].san}</span>`;
        if (history[i + 1]) {
            moveListHtml += `<span class="font-medium move-item p-1.5 rounded-md cursor-pointer transition-colors duration-150 hover:bg-tertiary" data-move-index="${i + 1}">${history[i+1].san}</span>`;
        }
        moveListHtml += `</div>`;
    }
    moveListHtml += '</div>';
    moveListContainer.html(moveListHtml);
    
    function updateState(moveIndex) {
        if (currentAudio) {
            currentAudio.pause();
        }
        
        currentMoveIndex = moveIndex;
        hideMoveQualityIcon(); 

        if (moveIndex !== lastAiCommentaryPly) {
             lastAiCommentaryPly = -1;
        }

        const newFen = fens[currentMoveIndex + 1];
        board.position(newFen, true);
        $('#white-clock-display').text(formatTime(whiteClockHistory[currentMoveIndex + 1]));
        $('#black-clock-display').text(formatTime(blackClockHistory[currentMoveIndex + 1]));
        $('.move-item').removeClass('move-active');
        const activeMoveEl = $(`.move-item[data-move-index=${currentMoveIndex}]`);
        if (activeMoveEl.length > 0) {
            activeMoveEl.addClass('move-active');
            activeMoveEl[0].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
        
        requestAnalysis(newFen, currentMoveIndex);
    }
    
    const boardConfig = {
        draggable: false,
        position: startFen || 'start',
        pieceTheme: function(piece) { return `/static/img/pieces/${piece.toLowerCase()}.png`; },
        moveSpeed: 200, 
    };
    board = Chessboard('board', boardConfig);
    
    const boardContainer = document.getElementById('board-container');
    new ResizeObserver(entries => {
        if (!entries || !entries.length) return;
        const { width, height } = entries[0].contentRect;
        const size = Math.min(width, height);
        const boardEl = document.getElementById('board');
        const overlayEl = document.getElementById('board-overlay'); 

        if (boardEl) {
            boardEl.style.width = `${size}px`;
            boardEl.style.height = `${size}px`;
        }
        if (overlayEl) { 
            overlayEl.style.width = `${size}px`;
            overlayEl.style.height = `${size}px`;
        }
        if (board) {
            board.resize();
            clearArrows();
            const currentFen = fens[currentMoveIndex + 1];
            if (currentFen) {
                requestAnalysis(currentFen, currentMoveIndex);
            }
        }
    }).observe(boardContainer);
    
    updateState(-1);
    
    $('#start-btn').on('click', () => updateState(-1));
    $('#end-btn').on('click', () => updateState(history.length - 1));
    $('#prev-btn').on('click', () => { if (currentMoveIndex > -1) updateState(currentMoveIndex - 1); });
    $('#next-btn').on('click', () => { if (currentMoveIndex < history.length - 1) updateState(currentMoveIndex + 1); });
    $('.move-item').on('click', function() { updateState(parseInt($(this).data('move-index'))); });
    
    $(document).on('keydown', function(e) {
        if ($('input:focus, button:focus').length > 0) return;
        if (e.key === "ArrowLeft") $('#prev-btn').click();
        else if (e.key === "ArrowRight") $('#next-btn').click();
        else if (e.key === "Home") { e.preventDefault(); $('#start-btn').click(); }
        else if (e.key === "End") { e.preventDefault(); $('#end-btn').click(); }
    });
    
    const $modal = $('#settings-modal');
    const $depthSlider = $('#depth-slider');
    const $threadsSlider = $('#threads-slider');
    const $depthValue = $('#depth-value');
    const $threadsValue = $('#threads-value');
    const $audioToggle = $('#audio-toggle');
    
    $('#stockfish-settings-btn').on('click', () => {
        $depthSlider.val(stockfishSettings.depth);
        $threadsSlider.val(stockfishSettings.threads);
        $depthValue.text(stockfishSettings.depth);
        $threadsValue.text(stockfishSettings.threads);
        $audioToggle.prop('checked', isAudioEnabled); 
        $modal.removeClass('hidden').addClass('flex');
    });
    
    function closeModal() {
        $modal.addClass('hidden').removeClass('flex');
    }
    
    $depthSlider.on('input', () => $depthValue.text($depthSlider.val()));
    $threadsSlider.on('input', () => $threadsValue.text($threadsSlider.val()));
    $('#settings-cancel-btn').on('click', closeModal);
    $modal.on('click', function(e) { if (e.target === this) closeModal(); });
    
    $('#settings-save-btn').on('click', () => {
        const oldDepth = stockfishSettings.depth;
        stockfishSettings.depth = parseInt($depthSlider.val());
        stockfishSettings.threads = parseInt($threadsSlider.val());
        isAudioEnabled = $audioToggle.is(':checked'); 

        if (!isAudioEnabled && currentAudio) {
            currentAudio.pause(); 
            currentAudio = null;
        }

        closeModal();
        
        // If analysis settings changed, clear cache and re-analyze current position
        if (oldDepth !== stockfishSettings.depth) {
            analysisCache = {};
            commentaryCache = {};
            requestAnalysis(fens[currentMoveIndex + 1], currentMoveIndex);
        }
    });
});