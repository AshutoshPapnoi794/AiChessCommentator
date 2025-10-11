// static/js/game.js

const moveQualityMap = {
    blunder: { icon: '/static/img/symbols/blunder.png', text: 'Blunder' },
    mistake: { icon: '/static/img/symbols/mistake.png', text: 'Mistake' },
    inaccuracy: { icon: '/static/img/symbols/inaccuracy.png', text: 'Inaccuracy' },
    good: { icon: '/static/img/symbols/good.png', text: 'Good Move' },
    excellent: { icon: '/static/img/symbols/excellent.png', text: 'Excellent Move' },
    best: { icon: '/static/img/symbols/best.png', text: 'Best Move' }
};

$(document).ready(function() {
    let socket = null;
    let analysisRequestCounter = 0;
    
    let isAiThinking = false;
    let lastAiCommentaryPly = -1;

    let currentAudio = null;
    let isAudioEnabled = false; 

    let moveQualityCache = {};
    let analysisCache = {};
    let commentaryCache = {};

    let stockfishSettings = { depth: 18, threads: 4 };
    
    const $arrowContainer = $('#arrow-container');
    
    function clearArrows() { $arrowContainer.empty(); }

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

        const fromCenter = { x: (fromRect.left - arrowContainerRect.left) + fromRect.width / 2, y: (fromRect.top - arrowContainerRect.top) + fromRect.height / 2 };
        const toCenter = { x: (toRect.left - arrowContainerRect.left) + toRect.width / 2, y: (toRect.top - arrowContainerRect.top) + toRect.height / 2 };

        const dx = toCenter.x - fromCenter.x;
        const dy = toCenter.y - fromCenter.y;
        const length = Math.sqrt(dx * dx + dy * dy);
        const angleRad = Math.atan2(dy, dx);
        
        const squareSize = boardRect.width / 8;
        const headWidth = squareSize * 0.5, headHeight = squareSize * 0.7, shaftHeight = squareSize * 0.30;
        const shaftLength = length - (headWidth * 0.8);

        const $wrapper = $('<div>').addClass('arrow-wrapper').css({ left: fromCenter.x, top: fromCenter.y, width: length, height: headHeight, transform: `rotate(${angleRad}rad)` });
        const $shaft = $('<div>').addClass('arrow-shaft').css({ height: shaftHeight, width: shaftLength > 0 ? shaftLength : 0 });
        const $head = $('<div>').addClass('arrow-head').css({ width: headWidth, height: headHeight });
        
        $wrapper.append($shaft, $head);
        $arrowContainer.append($wrapper);
    }

    function setAnalysisLoadingState(isLoading) {
        $('#analysis-loader').toggleClass('hidden', !isLoading);
        if (isLoading) $('#evaluation-score').text('...');
    }

    function setAiCommentaryLoading(isLoading) {
        isAiThinking = isLoading;
        if (isLoading) $('#ai-commentary').html('<span class="italic text-text-secondary">AI is thinking...</span>');
    }
    
    function displayMoveQualityIcon(classification, move) {
        const quality = moveQualityMap[classification];
        const $indicator = $('#move-quality-indicator');
        if (!quality || !move) return $indicator.addClass('opacity-0');

        const fileIndex = move.to.charCodeAt(0) - 'a'.charCodeAt(0);
        const rankIndex = 8 - parseInt(move.to.charAt(1));
        
        $('#move-quality-indicator').css({ left: (fileIndex * 12.5) + '%', top: (rankIndex * 12.5) + '%' });
        $('#move-quality-icon').attr('src', quality.icon);
        $('#move-quality-indicator').attr('title', quality.text).removeClass('opacity-0');
    }
    
    function hideMoveQualityIcon() { $('#move-quality-indicator').addClass('opacity-0'); }

    function displayCommentary(commentary) {
        const formatted = (commentary || 'Could not retrieve commentary.')
            .replace(/\*\*(.*?)\*\*/g, '<strong class="text-white font-semibold">$1</strong>')
            .replace(/\*(.*?)\*/g, '<em class="italic">$1</em>').replace(/\n/g, '<br>');
        $('#ai-commentary').html(formatted);
    }

    if (stockfishEnabled) {
        socket = io();

        socket.on('analysis_result', function(data) {
            if (data.requestId !== analysisRequestCounter) return;
            updateAnalysisUI(data);

            const isFinalDepth = data.depth >= stockfishSettings.depth;

            if (isFinalDepth) {
                setAnalysisLoadingState(false);
                analysisCache[currentMoveIndex] = data;

                if (commentaryCache[currentMoveIndex]) {
                    displayCommentary(commentaryCache[currentMoveIndex]);
                } else if (!isAiThinking && currentMoveIndex !== lastAiCommentaryPly) {
                    lastAiCommentaryPly = currentMoveIndex;
                    setAiCommentaryLoading(true);
                    
                    let pgn = '';
                    for (let i = 0; i <= currentMoveIndex; i++) {
                        if (i % 2 === 0) pgn += `${Math.floor(i / 2) + 1}. `;
                        pgn += `${history[i].san} `;
                    }

                    const commentaryPayload = {
                        pgn: pgn.trim(),
                        ply: currentMoveIndex + 1,
                        humanMove: history[currentMoveIndex].san,
                        evaluation: data.eval,
                        topLines: data.moves,
                        audio_enabled: isAudioEnabled,
                        current_fen: fens[currentMoveIndex + 1],
                        previous_fen: fens[currentMoveIndex]
                    };
                    socket.emit('get_ai_commentary', commentaryPayload);
                }
            }
        });

        socket.on('move_quality_result', function(data) {
            if (data.classification) {
                moveQualityCache[currentMoveIndex] = data.classification;
                displayMoveQualityIcon(data.classification, history[currentMoveIndex]);
            }
        });

        socket.on('analysis_error', function(data) {
            if (data.requestId && data.requestId !== analysisRequestCounter) return;
            setAnalysisLoadingState(false);
            clearArrows();
            console.error('Analysis Error:', data.message);
            $('#top-lines-display').html(`<p class="text-xs text-red-400 font-mono">${data.message}</p>`);
        });

        socket.on('ai_commentary_text_result', function(data) {
            setAiCommentaryLoading(false);
            commentaryCache[currentMoveIndex] = data.commentary;
            displayCommentary(data.commentary);
        });
        
        socket.on('ai_commentary_audio_result', function(data) {
            if (!isAudioEnabled) return; 
            if (currentAudio) currentAudio.pause();
            if (data.audio_data) {
                currentAudio = new Audio('data:audio/wav;base64,' + data.audio_data);
                currentAudio.play().catch(e => console.error("Audio playback failed:", e));
            }
        });
        
        socket.on('ai_commentary_error', function(data) {
            setAiCommentaryLoading(false);
            $('#ai-commentary').html(`<span class="text-xs text-red-400 font-mono">${data.message}</span>`);
        });

        socket.on('full_analysis_complete', function(data) {
            $('#header-accuracy-loading').hide();
            $('#header-accuracy-results').removeClass('hidden').addClass('flex');
            $('#accuracy-white').text(`${data.white}%`);
            $('#accuracy-black').text(`${data.black}%`);
        });
    }

    function requestAnalysis(fen, moveIndex) {
        if (!socket || !stockfishEnabled) return;

        if (analysisCache[moveIndex]) {
            setAnalysisLoadingState(false);
            updateAnalysisUI(analysisCache[moveIndex]);
            if (commentaryCache[moveIndex]) displayCommentary(commentaryCache[moveIndex]);
            if (moveQualityCache[moveIndex]) displayMoveQualityIcon(moveQualityCache[moveIndex], history[moveIndex]);
            return; 
        }

        $('#ai-commentary').html(moveIndex > -1 ? 
            '<span class="italic text-text-secondary">Waiting for engine analysis...</span>' : 
            '<span class="italic text-text-secondary">Navigate moves to see AI commentary.</span>');

        setAnalysisLoadingState(true);
        // We clear arrows in updateState now, but clearing here again is harmless and safe.
        clearArrows(); 
        socket.emit('analyze_position', { fen: fen, settings: stockfishSettings, requestId: ++analysisRequestCounter });

        if (moveIndex >= 0) {
            socket.emit('get_move_quality', { fen_before: fens[moveIndex], fen_after: fens[moveIndex + 1] });
        }
    }

    function updateAnalysisUI(data) {
        if (data.eval.type === 'cp') {
            const score = (data.eval.value / 100.0).toFixed(2);
            $('#evaluation-score').text(score > 0 ? `+${score}` : score);
            const percentage = 50 + (Math.max(-800, Math.min(800, data.eval.value)) / 800) * 50;
            $('#evaluation-bar-white').css('height', `${percentage}%`);
        } else if (data.eval.type === 'mate') {
            const mateIn = data.eval.value;
            $('#evaluation-score').text(mateIn > 0 ? `M${mateIn}` : `M-${Math.abs(mateIn)}`);
            $('#evaluation-bar-white').css('height', mateIn > 0 ? '100%' : '0%');
        }
        const $topLines = $('#top-lines-display');
        if (data.moves && data.moves.length > 0) {
            clearArrows();
            const topMove = data.moves[0];
            if (topMove && topMove.uci) drawArrow(topMove.uci.substring(0, 2), topMove.uci.substring(2, 4));
            
            let linesHtml = `<div class="w-full font-mono text-text-secondary"><div class="text-xs text-right pb-1 border-b border-border-primary/50 mb-1">Depth: ${data.depth}</div>`;
            data.moves.forEach((move, index) => {
                let scoreText = (move.mate !== null) ? `M${move.mate}` : (move.cp / 100.0).toFixed(2);
                if (move.cp > 0 && move.mate === null) scoreText = `+${scoreText}`;
                linesHtml += `<div class="flex justify-between items-center py-0.5"><span class="text-text-primary">${index + 1}. ${move.san}</span><span>${scoreText}</span></div>`;
            });
            $topLines.html(linesHtml + '</div>');
        } else {
            clearArrows();
            $topLines.html('<p class="text-xs text-text-secondary">No moves found.</p>');
        }
    }

    function formatTime(totalSeconds) {
        if (typeof totalSeconds !== 'number' || isNaN(totalSeconds)) return "00:00";
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = Math.floor(totalSeconds % 60);
        return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }

    const game = new Chess(startFen || undefined);
    const history = movesData ? movesData.map(m => game.move(m.san)).filter(Boolean) : [];
    game.reset();
    
    const fens = [startFen || new Chess().fen()];
    history.forEach(move => { game.move(move.san); fens.push(game.fen()); });
    game.reset();
    
    if (socket && stockfishEnabled) socket.emit('request_full_analysis', { fens: fens });
    
    const whiteClockHistory = [initialTimeSeconds], blackClockHistory = [initialTimeSeconds];
    let lastWClock = initialTimeSeconds, lastBClock = initialTimeSeconds;
    if (movesData) {
        movesData.forEach(move => {
            if (move.ply % 2 !== 0) lastWClock = move.clock; else lastBClock = move.clock;
            whiteClockHistory.push(lastWClock); blackClockHistory.push(lastBClock);
        });
    }
    
    let currentMoveIndex = -1;
    let board = null;
    let moveListHtml = '<div class="space-y-1">';
    for (let i = 0; i < history.length; i += 2) {
        moveListHtml += `<div class="grid grid-cols-[30px_1fr_1fr] gap-x-2 items-center text-base"><span class="text-text-secondary text-right">${i / 2 + 1}.</span><span class="font-medium move-item p-1.5 rounded-md cursor-pointer" data-move-index="${i}">${history[i].san}</span>`;
        if (history[i + 1]) moveListHtml += `<span class="font-medium move-item p-1.5 rounded-md cursor-pointer" data-move-index="${i + 1}">${history[i+1].san}</span>`;
        moveListHtml += `</div>`;
    }
    $('#move-list-container').html(moveListHtml + '</div>');
    
    function updateState(moveIndex) {
        if (currentAudio) {
            currentAudio.pause();
        }

        // --- BUG FIX: Clear arrows immediately on any state change ---
        clearArrows();
        
        currentMoveIndex = moveIndex;
        hideMoveQualityIcon(); 
        if (moveIndex !== lastAiCommentaryPly) lastAiCommentaryPly = -1;

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
        
        const $openingDisplay = $('#opening-name-display');
        const openingName = openingNamesByPly[currentMoveIndex + 1] || 'Unknown';
        $openingDisplay.text(openingName).attr('title', openingName);
        
        requestAnalysis(newFen, currentMoveIndex);
    }
    
    board = Chessboard('board', {
        draggable: false, position: startFen || 'start',
        pieceTheme: (p) => `/static/img/pieces/${p.toLowerCase()}.png`, moveSpeed: 200, 
    });
    
    new ResizeObserver(entries => {
        if (!entries || !entries.length) return;
        const size = Math.min(entries[0].contentRect.width, entries[0].contentRect.height);
        $('#board, #board-overlay').width(size).height(size);
        if (board) {
            board.resize();
            clearArrows();
            requestAnalysis(fens[currentMoveIndex + 1], currentMoveIndex);
        }
    }).observe(document.getElementById('board-container'));
    
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
    $('#stockfish-settings-btn').on('click', () => {
        $('#depth-slider').val(stockfishSettings.depth);
        $('#threads-slider').val(stockfishSettings.threads);
        $('#depth-value').text(stockfishSettings.depth);
        $('#threads-value').text(stockfishSettings.threads);
        $('#audio-toggle').prop('checked', isAudioEnabled); 
        $modal.removeClass('hidden').addClass('flex');
    });
    
    function closeModal() { $modal.addClass('hidden').removeClass('flex'); }
    
    $('#depth-slider').on('input', (e) => $('#depth-value').text(e.target.value));
    $('#threads-slider').on('input', (e) => $('#threads-value').text(e.target.value));
    $('#settings-cancel-btn, #settings-modal').on('click', function(e) { if (e.target === this) closeModal(); });
    
    $('#settings-save-btn').on('click', () => {
        const oldDepth = stockfishSettings.depth;
        stockfishSettings.depth = parseInt($('#depth-slider').val());
        stockfishSettings.threads = parseInt($('#threads-slider').val());
        isAudioEnabled = $('#audio-toggle').is(':checked'); 
        if (!isAudioEnabled && currentAudio) { currentAudio.pause(); currentAudio = null; }

        closeModal();
        if (oldDepth !== stockfishSettings.depth) {
            analysisCache = {}; commentaryCache = {};
            requestAnalysis(fens[currentMoveIndex + 1], currentMoveIndex);
        }
    });
});