(() => {
    const moveQualityMap = {
        blunder: { icon: '/static/img/symbols/blunder.png', text: 'Blunder' },
        mistake: { icon: '/static/img/symbols/mistake.png', text: 'Mistake' },
        inaccuracy: { icon: '/static/img/symbols/inaccuracy.png', text: 'Inaccuracy' },
        good: { icon: '/static/img/symbols/good.png', text: 'Good Move' },
        excellent: { icon: '/static/img/symbols/excellent.png', text: 'Excellent Move' },
        best: { icon: '/static/img/symbols/best.png', text: 'Best Move' },
        brilliant: { icon: '/static/img/symbols/brilliant.png', text: 'Brilliant Move' },
        great: { icon: '/static/img/symbols/great.png', text: 'Great Move' },
        miss: { icon: '/static/img/symbols/miss.png', text: 'Miss' },
        book: { icon: '/static/img/symbols/book.png', text: 'Book Move' },
    };
    const SILENT_WAV_DATA_URI = 'data:audio/wav;base64,UklGRgQCAABXQVZFZm10IBAAAAABAAEAwF0AAIC7AAACABAAZGF0YeABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=';

    const data = window.GAME_DATA || {};
    if (typeof Chess === 'undefined' || typeof Chessboard === 'undefined') {
        const commentaryEl = document.getElementById('ai-commentary');
        if (commentaryEl) {
            commentaryEl.textContent = 'Failed to load board libraries. Refresh the page.';
        }
        return;
    }

    const START_FEN = data.startFen || 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    const history = Array.isArray(data.moves) ? data.moves : [];
    const openings = Array.isArray(data.openings) ? data.openings : [];
    const stockfishEnabled = Boolean(data.stockfish);
    const initialTime = Number.isFinite(Number(data.initialTime)) ? Number(data.initialTime) : 600;
    const initialTacticsEnabled = typeof data.initialTacticsEnabled === 'boolean' ? data.initialTacticsEnabled : true;

    const state = {
        currentPly: 0,
        socket: null,
        audioEnabled: false,
        currentAudio: null,
        audioPrimed: false,
        commentaryCache: new Map(),
        batchCommentaryReady: false,
        batchCommentaryInFlight: false,
        batchCommentaryFailed: false,
        batchPrefetchAttempted: false,
        batchGameKey: '',
        commentaryInFlightForPly: null,
        analysisSettings: {
            depth: 16,
            threads: 1,
            tacticsEnabled: initialTacticsEnabled,
        },
        currentAnalysisRequestId: 0,
        typewriterTimer: null,
        reduceMotion: window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches,
        arrow: null,
        moveQualityCache: new Map(),
    };

    const el = {
        aiCommentary: document.getElementById('ai-commentary'),
        evalBarFill: document.getElementById('eval-bar-fill'),
        evalScore: document.getElementById('eval-score'),
        topLines: document.getElementById('top-lines'),
        openingName: document.getElementById('opening-name'),
        moveList: document.getElementById('move-list'),
        arrowLayer: document.getElementById('arrow-layer'),
        boardContainer: document.getElementById('board-container'),
        clockWhite: document.getElementById('clock-white'),
        clockBlack: document.getElementById('clock-black'),
        engineStatus: document.getElementById('engine-status'),
        depthChip: document.getElementById('depth-chip'),
        depthSlider: document.getElementById('depth-slider'),
        depthVal: document.getElementById('depth-val'),
        tacticsToggle: document.getElementById('tactics-toggle'),
        audioToggle: document.getElementById('audio-toggle'),
        settingsBtn: document.getElementById('settings-btn'),
        settingsModal: document.getElementById('settings-modal'),
        modalClose: document.getElementById('modal-close'),
        modalSave: document.getElementById('modal-save'),
        btnStart: document.getElementById('btn-start'),
        btnPrev: document.getElementById('btn-prev'),
        btnNext: document.getElementById('btn-next'),
        btnEnd: document.getElementById('btn-end'),
        moveQualityIndicator: document.getElementById('move-quality-indicator'),
        moveQualityIcon: document.getElementById('move-quality-icon'),
        batchProgressWrap: document.getElementById('batch-progress-wrap'),
        batchProgressFill: document.getElementById('batch-progress-fill'),
        batchProgressLabel: document.getElementById('batch-progress-label'),
        batchProgressPct: document.getElementById('batch-progress-pct'),
    };

    const { fens, moveSquares } = buildFenHistoryAndSquares(START_FEN, history);
    const clockSnapshots = buildClockSnapshots(history, initialTime);
    state.batchGameKey = computeGameKey();

    const board = Chessboard('board', {
        position: fens[0],
        draggable: false,
        pieceTheme(piece) {
            return `/static/img/pieces/${piece.toLowerCase()}.png`;
        },
    });

    buildMoveList();
    bindNavigation();
    bindSettings();
    setupBoardResizeObserver();
    connectSocket();
    goTo(0, false);

    function buildFenHistoryAndSquares(startFen, moves) {
        const replay = new Chess();
        try {
            replay.load(startFen);
        } catch (error) {
            replay.reset();
        }

        const fenHistory = [replay.fen()];
        const squares = [];
        for (const move of moves) {
            if (!move || !move.san) {
                continue;
            }
            const madeMove = replay.move(move.san, { sloppy: true });
            if (!madeMove) {
                break;
            }
            squares.push({
                from: madeMove.from,
                to: madeMove.to,
            });
            fenHistory.push(replay.fen());
        }
        return { fens: fenHistory, moveSquares: squares };
    }

    function buildClockSnapshots(moves, startSeconds) {
        let whiteSeconds = startSeconds;
        let blackSeconds = startSeconds;
        const snapshots = [{ white: whiteSeconds, black: blackSeconds }];

        moves.forEach((move, idx) => {
            const moveClock = Number(move.clock);
            if (Number.isFinite(moveClock)) {
                if (idx % 2 === 0) {
                    whiteSeconds = moveClock;
                } else {
                    blackSeconds = moveClock;
                }
            }
            snapshots.push({ white: whiteSeconds, black: blackSeconds });
        });

        return snapshots;
    }

    function buildMoveList() {
        if (!el.moveList) {
            return;
        }

        const fragment = document.createDocumentFragment();
        for (let i = 0; i < history.length; i += 2) {
            const row = document.createElement('div');
            row.className = 'grid grid-cols-[32px_1fr_1fr] gap-1 rounded-lg px-1 py-1 hover:bg-white/5 transition-colors';

            const moveNumber = document.createElement('span');
            moveNumber.className = 'text-slate-500 text-right pr-2 font-mono text-xs';
            moveNumber.textContent = `${Math.floor(i / 2) + 1}.`;
            row.appendChild(moveNumber);

            const whiteMove = document.createElement('button');
            whiteMove.type = 'button';
            whiteMove.className = 'move-row text-left px-2 py-1 rounded text-slate-200 hover:bg-slate-800/70 transition-colors';
            whiteMove.dataset.ply = String(i + 1);
            whiteMove.textContent = history[i].san;
            row.appendChild(whiteMove);

            const blackMove = document.createElement('button');
            blackMove.type = 'button';
            blackMove.className = 'move-row text-left px-2 py-1 rounded text-slate-200 hover:bg-slate-800/70 transition-colors';
            blackMove.dataset.ply = String(i + 2);
            if (history[i + 1]) {
                blackMove.textContent = history[i + 1].san;
            } else {
                blackMove.disabled = true;
                blackMove.classList.add('opacity-40', 'cursor-default');
                blackMove.textContent = '...';
            }
            row.appendChild(blackMove);
            fragment.appendChild(row);
        }

        el.moveList.innerHTML = '';
        el.moveList.appendChild(fragment);

        el.moveList.addEventListener('click', (event) => {
            const target = event.target;
            if (!(target instanceof HTMLElement)) {
                return;
            }
            const moveButton = target.closest('.move-row');
            if (!moveButton || !(moveButton instanceof HTMLElement)) {
                return;
            }
            if (moveButton.dataset.ply) {
                goTo(Number(moveButton.dataset.ply), true);
            }
        });
    }

    function bindNavigation() {
        if (el.btnStart) {
            el.btnStart.addEventListener('click', () => goTo(0, true));
        }
        if (el.btnPrev) {
            el.btnPrev.addEventListener('click', () => goTo(state.currentPly - 1, true));
        }
        if (el.btnNext) {
            el.btnNext.addEventListener('click', () => goTo(state.currentPly + 1, true));
        }
        if (el.btnEnd) {
            el.btnEnd.addEventListener('click', () => goTo(history.length, true));
        }

        document.addEventListener('keydown', (event) => {
            if (event.target instanceof HTMLElement && event.target.closest('input, textarea, select')) {
                return;
            }
            if (el.settingsModal && !el.settingsModal.classList.contains('hidden')) {
                if (event.key === 'Escape') {
                    toggleSettingsModal(false);
                }
                return;
            }

            if (event.key === 'ArrowLeft') {
                goTo(state.currentPly - 1, true);
            } else if (event.key === 'ArrowRight') {
                goTo(state.currentPly + 1, true);
            } else if (event.key === 'ArrowUp') {
                goTo(0, true);
            } else if (event.key === 'ArrowDown') {
                goTo(history.length, true);
            }
        });
    }

    function bindSettings() {
        if (!el.depthSlider || !el.depthVal || !el.depthChip) {
            return;
        }

        el.depthSlider.value = String(state.analysisSettings.depth);
        el.depthVal.textContent = String(state.analysisSettings.depth);
        el.depthChip.textContent = `D${state.analysisSettings.depth}`;
        if (el.tacticsToggle) {
            el.tacticsToggle.checked = state.analysisSettings.tacticsEnabled;
        }
        if (el.audioToggle) {
            el.audioToggle.checked = state.audioEnabled;
        }

        el.depthSlider.addEventListener('input', () => {
            const depth = clampInt(Number(el.depthSlider.value), 10, 22, 16);
            el.depthVal.textContent = String(depth);
        });

        if (el.settingsBtn) {
            el.settingsBtn.addEventListener('click', () => toggleSettingsModal(true));
        }
        if (el.modalClose) {
            el.modalClose.addEventListener('click', () => toggleSettingsModal(false));
        }
        if (el.modalSave) {
            el.modalSave.addEventListener('click', () => {
                const tacticsChanged = el.tacticsToggle
                    ? state.analysisSettings.tacticsEnabled !== el.tacticsToggle.checked
                    : false;
                const audioWasEnabled = state.audioEnabled;
                state.analysisSettings.depth = clampInt(Number(el.depthSlider.value), 10, 22, 16);
                el.depthChip.textContent = `D${state.analysisSettings.depth}`;
                if (el.tacticsToggle) {
                    state.analysisSettings.tacticsEnabled = el.tacticsToggle.checked;
                }
                if (el.audioToggle) {
                    state.audioEnabled = el.audioToggle.checked;
                    if (!state.audioEnabled) {
                        stopNarration();
                        state.audioPrimed = false;
                    } else if (!audioWasEnabled && state.currentPly > 0 && !tacticsChanged) {
                        primeAudioPlayback();
                        const currentCommentary = state.commentaryCache.get(state.currentPly);
                        if (currentCommentary) {
                            requestCommentaryAudio(currentCommentary, state.currentPly);
                        }
                    }
                }
                toggleSettingsModal(false);
                if (tacticsChanged) {
                    state.batchGameKey = computeGameKey();
                    resetDerivedAnalysisState();
                    if (state.currentPly > 0) {
                        renderCommentary('Refreshing analysis...');
                    }
                }
                if (stockfishEnabled && state.currentPly > 0) {
                    requestAnalysisForCurrentPosition();
                    if (tacticsChanged) {
                        requestCommentary();
                    }
                }
            });
        }

        if (el.settingsModal) {
            el.settingsModal.addEventListener('click', (event) => {
                if (event.target === el.settingsModal) {
                    toggleSettingsModal(false);
                }
            });
        }
    }

    function toggleSettingsModal(open) {
        if (!el.settingsModal) {
            return;
        }
        if (open) {
            el.settingsModal.classList.remove('hidden');
            el.settingsModal.classList.add('flex');
            return;
        }
        el.settingsModal.classList.add('hidden');
        el.settingsModal.classList.remove('flex');
    }

    function setupBoardResizeObserver() {
        if (!el.boardContainer) {
            return;
        }

        const resize = () => {
            board.resize();
            redrawArrow();
            if (state.currentPly > 0) {
                const quality = state.moveQualityCache.get(state.currentPly);
                if (quality) {
                    showMoveQualityIcon(state.currentPly, quality);
                } else {
                    hideMoveQualityIcon();
                }
            } else {
                hideMoveQualityIcon();
            }
        };

        if (typeof ResizeObserver !== 'undefined') {
            const observer = new ResizeObserver(resize);
            observer.observe(el.boardContainer);
        }

        window.addEventListener('resize', resize);
        window.addEventListener('orientationchange', resize);

        window.requestAnimationFrame(() => {
            resize();
        });
    }

    function connectSocket() {
        if (!stockfishEnabled || typeof io === 'undefined') {
            setEngineStatus('Engine Offline');
            return;
        }

        state.socket = io();

        state.socket.on('connect', () => {
            setEngineStatus('Engine Connected');
        });

        state.socket.on('connect_error', () => {
            setEngineStatus('Engine Error');
        });

        state.socket.on('disconnect', () => {
            setEngineStatus('Engine Reconnecting...');
            if (!state.batchCommentaryReady) {
                state.batchPrefetchAttempted = false;
                state.batchCommentaryInFlight = false;
            }
        });

        state.socket.on('analysis_result', (result) => {
            if (!result || typeof result !== 'object') {
                return;
            }
            if (typeof result.requestId === 'number' && result.requestId < state.currentAnalysisRequestId) {
                return;
            }
            if (result.fen && result.fen !== fens[state.currentPly]) {
                return;
            }

            renderEvaluation(result.eval);
            renderTopLines(result.moves);

            if (Array.isArray(result.moves) && result.moves.length > 0 && result.moves[0].uci) {
                const bestMove = result.moves[0].uci;
                state.arrow = {
                    from: bestMove.slice(0, 2),
                    to: bestMove.slice(2, 4),
                };
                redrawArrow();
            } else {
                state.arrow = null;
                clearArrow();
            }

            if (state.currentPly > 0 && !state.commentaryCache.has(state.currentPly) && result.depth >= 12) {
                if (!state.batchCommentaryInFlight && !state.batchCommentaryReady && state.batchCommentaryFailed) {
                    requestCommentary();
                }
            }
        });

        state.socket.on('batch_commentary_progress', (payload) => {
            if (!payload || typeof payload !== 'object') {
                return;
            }
            if (payload.gameKey && String(payload.gameKey) !== state.batchGameKey) {
                return;
            }
            const pctRaw = Number(payload.progress);
            const pct = Number.isFinite(pctRaw) ? Math.max(0, Math.min(100, Math.round(pctRaw))) : 0;
            const message = payload.message ? String(payload.message) : 'Preparing full-game commentary...';
            showBatchProgress(pct, message);
            state.batchCommentaryInFlight = pct < 100;
            if (pct >= 100) {
                hideBatchProgress();
            }
        });

        state.socket.on('batch_commentary_ready', (payload) => {
            if (!payload || typeof payload !== 'object') {
                return;
            }
            if (payload.gameKey && String(payload.gameKey) !== state.batchGameKey) {
                return;
            }
            const commentaryMap = payload.commentaries && typeof payload.commentaries === 'object'
                ? payload.commentaries
                : {};
            for (const [plyKey, text] of Object.entries(commentaryMap)) {
                const ply = Number(plyKey);
                if (!Number.isFinite(ply) || ply <= 0 || !text) {
                    continue;
                }
                state.commentaryCache.set(ply, String(text));
            }

            const qualityMap = payload.move_qualities && typeof payload.move_qualities === 'object'
                ? payload.move_qualities
                : {};
            for (const [plyKey, quality] of Object.entries(qualityMap)) {
                const ply = Number(plyKey);
                if (!Number.isFinite(ply) || ply <= 0 || !quality) {
                    continue;
                }
                state.moveQualityCache.set(ply, String(quality).toLowerCase());
            }

            state.batchCommentaryReady = true;
            state.batchCommentaryInFlight = false;
            state.batchCommentaryFailed = false;
            hideBatchProgress();

            if (state.currentPly > 0 && state.commentaryCache.has(state.currentPly)) {
                const currentText = state.commentaryCache.get(state.currentPly);
                renderCommentary(currentText);
                requestCommentaryAudio(currentText, state.currentPly);
            }
            if (state.currentPly > 0 && state.moveQualityCache.has(state.currentPly)) {
                showMoveQualityIcon(state.currentPly, state.moveQualityCache.get(state.currentPly));
            }
        });

        state.socket.on('batch_commentary_error', (payload) => {
            if (!payload || typeof payload !== 'object') {
                return;
            }
            if (payload.gameKey && String(payload.gameKey) !== state.batchGameKey) {
                return;
            }
            state.batchCommentaryInFlight = false;
            state.batchCommentaryFailed = true;
            hideBatchProgress();
            if (state.currentPly > 0 && !state.commentaryCache.has(state.currentPly)) {
                renderCommentary('Batch commentary failed. Falling back to per-move analysis...');
                requestCommentary();
            }
        });

        state.socket.on('ai_commentary_text_result', (payload) => {
            const commentary = payload && payload.commentary ? String(payload.commentary) : '';
            const ply = payload && Number.isFinite(Number(payload.ply)) ? Number(payload.ply) : state.currentPly;
            if (!commentary) {
                return;
            }

            state.commentaryCache.set(ply, commentary);
            const quality = payload && payload.quality ? String(payload.quality).toLowerCase() : '';
            if (quality && moveQualityMap[quality]) {
                state.moveQualityCache.set(ply, quality);
            }
            if (state.commentaryInFlightForPly === ply) {
                state.commentaryInFlightForPly = null;
            }
            if (ply === state.currentPly) {
                renderCommentary(commentary);
                if (quality && moveQualityMap[quality]) {
                    showMoveQualityIcon(ply, quality);
                }
            }
        });

        state.socket.on('ai_commentary_audio_result', (payload) => {
            if (!state.audioEnabled || !payload || !payload.audio_data) {
                return;
            }
            const ply = Number(payload.ply);
            if (Number.isFinite(ply) && ply !== state.currentPly) {
                return;
            }

            stopCurrentAudioPlayback();

            try {
                const mimeType = payload.audio_mime_type ? String(payload.audio_mime_type) : 'audio/wav';
                state.currentAudio = new Audio(`data:${mimeType};base64,${payload.audio_data}`);
                state.currentAudio.play().catch((error) => {
                    // Browser autoplay policy may block playback.
                    console.warn('Audio playback failed:', error);
                });
            } catch (error) {
                console.warn('Malformed audio payload received:', error);
            }
        });

        state.socket.on('ai_commentary_audio_error', (payload) => {
            const ply = payload && Number.isFinite(Number(payload.ply)) ? Number(payload.ply) : state.currentPly;
            if (ply !== state.currentPly) {
                return;
            }
            const message = payload && payload.message ? String(payload.message) : 'TTS audio generation failed.';
            console.warn(message);
        });

        state.socket.on('ai_commentary_error', (payload) => {
            const ply = payload && Number.isFinite(Number(payload.ply)) ? Number(payload.ply) : state.currentPly;
            if (ply !== state.currentPly) {
                return;
            }
            const message = payload && payload.message ? String(payload.message) : 'Commentary unavailable.';
            renderCommentary(message);
            state.commentaryInFlightForPly = null;
        });

        state.socket.on('move_quality_result', (payload) => {
            if (!payload || typeof payload !== 'object') {
                return;
            }
            if (typeof payload.requestId === 'number' && payload.requestId < state.currentAnalysisRequestId) {
                return;
            }
            const ply = Number(payload.ply);
            if (!Number.isFinite(ply) || ply <= 0) {
                return;
            }
            const classification = payload.classification ? String(payload.classification).toLowerCase() : '';
            if (!classification || !moveQualityMap[classification]) {
                return;
            }
            state.moveQualityCache.set(ply, classification);
            if (ply === state.currentPly) {
                showMoveQualityIcon(ply, classification);
            }
        });
    }

    function goTo(nextPly, userInitiated) {
        const maxPly = Math.min(history.length, fens.length - 1);
        const clampedPly = clampInt(nextPly, 0, maxPly, 0);
        if (userInitiated && state.audioEnabled) {
            primeAudioPlayback();
        }
        stopNarration();
        state.currentPly = clampedPly;

        board.position(fens[clampedPly], false);
        highlightCurrentMove(clampedPly, userInitiated);

        if (el.openingName) {
            el.openingName.textContent = openings[clampedPly] || 'Unknown';
        }

        updateClockDisplay(clampedPly);
        state.arrow = null;
        clearArrow();
        hideMoveQualityIcon();

        if (clampedPly === 0) {
            renderCommentary('Game start. Use the move list, buttons, or arrow keys to navigate.');
            if (el.evalScore) {
                el.evalScore.textContent = '0.00';
            }
            if (el.evalBarFill) {
                el.evalBarFill.style.height = '50%';
            }
            if (el.topLines) {
                el.topLines.innerHTML = '<span class="text-slate-500 text-xs italic">Waiting for engine...</span>';
            }
            return;
        }

        if (!stockfishEnabled || !state.socket) {
            renderCommentary('Engine analysis is unavailable for this game.');
            return;
        }

        requestAnalysisForCurrentPosition();

        const cachedQuality = state.moveQualityCache.get(clampedPly);
        if (cachedQuality) {
            showMoveQualityIcon(clampedPly, cachedQuality);
        }

        const cachedCommentary = state.commentaryCache.get(clampedPly);
        if (cachedCommentary) {
            renderCommentary(cachedCommentary);
            requestCommentaryAudio(cachedCommentary, clampedPly);
        } else {
            renderCommentary('Thinking...');
            requestCommentary();
        }
    }

    function highlightCurrentMove(ply, userInitiated) {
        const rows = document.querySelectorAll('.move-row');
        rows.forEach((node) => {
            node.classList.remove('move-active');
        });

        if (ply === 0) {
            return;
        }

        const currentMove = document.querySelector(`.move-row[data-ply="${ply}"]`);
        if (!currentMove) {
            return;
        }

        currentMove.classList.add('move-active');
        if (userInitiated) {
            currentMove.scrollIntoView({ block: 'center', behavior: state.reduceMotion ? 'auto' : 'smooth' });
        }
    }

    function requestAnalysisForCurrentPosition() {
        if (!state.socket || state.currentPly <= 0) {
            return;
        }

        state.currentAnalysisRequestId += 1;
        const requestId = state.currentAnalysisRequestId;
        const fen = fens[state.currentPly];
        state.socket.emit('analyze_position', {
            fen,
            settings: state.analysisSettings,
            requestId,
        });
        if (state.currentPly > 0 && !state.moveQualityCache.has(state.currentPly)) {
            state.socket.emit('get_move_quality', {
                fen_before: fens[state.currentPly - 1],
                fen_after: fens[state.currentPly],
                ply: state.currentPly,
                requestId,
            });
        }

        if (el.topLines) {
            el.topLines.innerHTML = '<span class="text-slate-500 text-xs italic">Evaluating...</span>';
        }
        if (el.depthChip) {
            el.depthChip.textContent = `D${state.analysisSettings.depth}`;
        }
    }

    function requestCommentary() {
        if (!state.socket || state.currentPly <= 0) {
            return;
        }
        if (state.commentaryCache.has(state.currentPly)) {
            renderCommentary(state.commentaryCache.get(state.currentPly));
            return;
        }
        if (state.commentaryInFlightForPly === state.currentPly) {
            return;
        }

        state.commentaryInFlightForPly = state.currentPly;
        const move = history[state.currentPly - 1];
        if (!move || !move.san) {
            state.commentaryInFlightForPly = null;
            return;
        }

        state.socket.emit('get_ai_commentary', {
            ply: state.currentPly,
            humanMove: move.san,
            current_fen: fens[state.currentPly],
            previous_fen: fens[state.currentPly - 1],
            opening: openings[state.currentPly] || 'Unknown',
            audio_enabled: state.audioEnabled,
            settings: state.analysisSettings,
        });
    }

    function requestCommentaryAudio(commentary, ply) {
        if (!state.socket || !state.audioEnabled || !commentary) {
            return;
        }
        if (!Number.isFinite(ply) || ply <= 0) {
            return;
        }
        state.socket.emit('synthesize_commentary_audio', {
            ply,
            commentary: String(commentary),
        });
    }

    function requestBatchCommentaryPrefetch(force) {
        if (!state.socket || history.length === 0) {
            return;
        }
        if (!state.analysisSettings.tacticsEnabled) {
            state.batchCommentaryInFlight = false;
            hideBatchProgress();
            return;
        }
        if (!force && (state.batchPrefetchAttempted || state.batchCommentaryInFlight || state.batchCommentaryReady)) {
            return;
        }

        state.batchPrefetchAttempted = true;
        state.batchCommentaryInFlight = true;
        state.batchCommentaryFailed = false;
        showBatchProgress(0, 'Preparing full-game commentary...');

        const sanMoves = history.map((move) => (move && move.san ? String(move.san) : ''));
        state.socket.emit('prefetch_ai_commentary_batch', {
            gameKey: state.batchGameKey,
            fens,
            moves: sanMoves,
            openings,
            force: Boolean(force),
            settings: state.analysisSettings,
        });
    }

    function showBatchProgress(pct, label) {
        if (el.batchProgressWrap) {
            el.batchProgressWrap.classList.remove('hidden');
        }
        if (el.batchProgressFill) {
            el.batchProgressFill.style.width = `${Math.max(0, Math.min(100, Number(pct) || 0))}%`;
        }
        if (el.batchProgressLabel && label) {
            el.batchProgressLabel.textContent = String(label);
        }
        if (el.batchProgressPct) {
            const shownPct = Math.max(0, Math.min(100, Math.round(Number(pct) || 0)));
            el.batchProgressPct.textContent = `${shownPct}%`;
        }
    }

    function hideBatchProgress() {
        if (el.batchProgressWrap) {
            el.batchProgressWrap.classList.add('hidden');
        }
    }

    function resetDerivedAnalysisState() {
        state.commentaryCache.clear();
        state.moveQualityCache.clear();
        state.commentaryInFlightForPly = null;
        state.batchCommentaryReady = false;
        state.batchCommentaryInFlight = false;
        state.batchCommentaryFailed = false;
        state.batchPrefetchAttempted = false;
        stopNarration();
        hideBatchProgress();
        hideMoveQualityIcon();
    }

    function renderEvaluation(evalData) {
        let score = 0;
        if (!evalData || typeof evalData !== 'object') {
            if (el.evalScore) {
                el.evalScore.textContent = '0.00';
            }
            if (el.evalBarFill) {
                el.evalBarFill.style.height = '50%';
            }
            return;
        }

        if (evalData.type === 'mate') {
            const mateValue = Number(evalData.value) || 0;
            score = mateValue > 0 ? 10 : -10;
            if (el.evalScore) {
                el.evalScore.textContent = mateValue > 0 ? `M${mateValue}` : `M${Math.abs(mateValue)}`;
            }
        } else {
            score = (Number(evalData.value) || 0) / 100;
            if (el.evalScore) {
                const sign = score > 0 ? '+' : '';
                el.evalScore.textContent = `${sign}${score.toFixed(2)}`;
            }
        }

        const capped = Math.max(-5, Math.min(5, score));
        const percent = 50 + capped * 10;
        if (el.evalBarFill) {
            el.evalBarFill.style.height = `${percent}%`;
        }
    }

    function renderTopLines(moves) {
        if (!el.topLines) {
            return;
        }
        if (!Array.isArray(moves) || moves.length === 0) {
            el.topLines.innerHTML = '<span class="text-slate-500 text-xs italic">No principal variation yet.</span>';
            return;
        }

        const html = moves.map((move, idx) => {
            const mate = Number(move.mate);
            const cp = Number(move.cp);
            let scoreText = '--';
            let positive = false;

            if (Number.isFinite(mate) && mate !== 0) {
                scoreText = `M${Math.abs(mate)}`;
                positive = mate > 0;
            } else if (Number.isFinite(cp)) {
                const cpScore = cp / 100;
                scoreText = `${cpScore > 0 ? '+' : ''}${cpScore.toFixed(2)}`;
                positive = cpScore > 0;
            }

            const scoreColor = positive ? 'text-emerald-300' : 'text-rose-300';
            const moveSan = move.san || move.uci || '--';
            return `
                <div class="flex items-center justify-between text-xs px-1 py-1 border-b border-white/5 last:border-b-0">
                    <span class="text-slate-300">${idx + 1}. <span class="font-semibold text-slate-100">${moveSan}</span></span>
                    <span class="${scoreColor} font-semibold">${scoreText}</span>
                </div>
            `;
        }).join('');

        el.topLines.innerHTML = html;
    }

    function renderCommentary(text) {
        if (!el.aiCommentary) {
            return;
        }

        clearTypewriter();
        if (state.reduceMotion || text.length < 6) {
            el.aiCommentary.textContent = text;
            return;
        }

        el.aiCommentary.textContent = '';
        let cursor = 0;
        state.typewriterTimer = window.setInterval(() => {
            cursor += 2;
            el.aiCommentary.textContent = text.slice(0, cursor);
            if (cursor >= text.length) {
                clearTypewriter();
            }
        }, 16);
    }

    function clearTypewriter() {
        if (state.typewriterTimer) {
            window.clearInterval(state.typewriterTimer);
            state.typewriterTimer = null;
        }
    }

    function redrawArrow() {
        if (!state.arrow) {
            clearArrow();
            return;
        }
        drawArrow(state.arrow.from, state.arrow.to);
    }

    function drawArrow(from, to) {
        if (!el.arrowLayer) {
            return;
        }

        const boardElement = document.getElementById('board');
        if (!boardElement) {
            return;
        }

        const fromSquare = boardElement.querySelector(`.square-${from}`);
        const toSquare = boardElement.querySelector(`.square-${to}`);
        if (!fromSquare || !toSquare) {
            clearArrow();
            return;
        }

        const boardRect = boardElement.getBoundingClientRect();
        const fromRect = fromSquare.getBoundingClientRect();
        const toRect = toSquare.getBoundingClientRect();

        const startX = fromRect.left - boardRect.left + fromRect.width / 2;
        const startY = fromRect.top - boardRect.top + fromRect.height / 2;
        const endXRaw = toRect.left - boardRect.left + toRect.width / 2;
        const endYRaw = toRect.top - boardRect.top + toRect.height / 2;

        const dx = endXRaw - startX;
        const dy = endYRaw - startY;
        const length = Math.sqrt(dx * dx + dy * dy);
        if (!length) {
            clearArrow();
            return;
        }

        const tipPadding = 18;
        const endX = endXRaw - (dx / length) * tipPadding;
        const endY = endYRaw - (dy / length) * tipPadding;

        el.arrowLayer.innerHTML = `
            <svg width="100%" height="100%" viewBox="0 0 ${boardRect.width} ${boardRect.height}" preserveAspectRatio="none">
                <defs>
                    <marker id="analysis-arrow-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                        <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(245, 158, 11, 0.85)"></path>
                    </marker>
                </defs>
                <line
                    x1="${startX}"
                    y1="${startY}"
                    x2="${endX}"
                    y2="${endY}"
                    stroke="rgba(245, 158, 11, 0.85)"
                    stroke-width="8"
                    stroke-linecap="round"
                    marker-end="url(#analysis-arrow-head)"
                />
            </svg>
        `;
    }

    function clearArrow() {
        if (el.arrowLayer) {
            el.arrowLayer.innerHTML = '';
        }
    }

    function showMoveQualityIcon(ply, classification) {
        if (!el.moveQualityIndicator || !el.moveQualityIcon || !el.boardContainer) {
            return;
        }
        const quality = moveQualityMap[classification];
        const move = moveSquares[ply - 1];
        if (!quality || !move || !move.to) {
            hideMoveQualityIcon();
            return;
        }

        const boardElement = document.getElementById('board');
        const squareElement = boardElement ? boardElement.querySelector(`.square-${move.to}`) : null;
        if (!squareElement) {
            hideMoveQualityIcon();
            return;
        }

        const containerRect = el.boardContainer.getBoundingClientRect();
        const squareRect = squareElement.getBoundingClientRect();
        const inset = 2;

        el.moveQualityIcon.src = quality.icon;
        el.moveQualityIcon.alt = quality.text;
        el.moveQualityIndicator.title = quality.text;
        el.moveQualityIndicator.style.left = `${squareRect.right - containerRect.left - inset}px`;
        el.moveQualityIndicator.style.top = `${squareRect.top - containerRect.top + inset}px`;
        el.moveQualityIndicator.classList.remove('opacity-0');
    }

    function hideMoveQualityIcon() {
        if (el.moveQualityIndicator) {
            el.moveQualityIndicator.classList.add('opacity-0');
        }
    }

    function updateClockDisplay(ply) {
        const snapshot = clockSnapshots[ply] || clockSnapshots[clockSnapshots.length - 1];
        if (!snapshot) {
            return;
        }
        if (el.clockWhite) {
            el.clockWhite.textContent = formatClock(snapshot.white);
        }
        if (el.clockBlack) {
            el.clockBlack.textContent = formatClock(snapshot.black);
        }
    }

    function formatClock(secondsRaw) {
        const seconds = Number(secondsRaw);
        if (!Number.isFinite(seconds)) {
            return '--:--';
        }
        const total = Math.max(0, Math.floor(seconds));
        const hours = Math.floor(total / 3600);
        const mins = Math.floor((total % 3600) / 60);
        const secs = total % 60;

        if (hours > 0) {
            return `${hours}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
        }
        return `${mins}:${String(secs).padStart(2, '0')}`;
    }

    function stopCurrentAudioPlayback() {
        if (state.currentAudio) {
            state.currentAudio.pause();
            state.currentAudio = null;
        }
    }

    function primeAudioPlayback() {
        if (!state.audioEnabled || state.audioPrimed) {
            return;
        }
        try {
            const probe = new Audio(SILENT_WAV_DATA_URI);
            probe.muted = true;
            probe.play()
                .then(() => {
                    probe.pause();
                    state.audioPrimed = true;
                })
                .catch(() => {
                    // Keep retrying on next user interaction.
                });
        } catch (error) {
            // Ignore priming failures.
        }
    }

    function stopNarration() {
        stopCurrentAudioPlayback();
    }

    function setEngineStatus(text) {
        if (el.engineStatus) {
            el.engineStatus.textContent = text;
        }
    }

    function computeGameKey() {
        const seed = `${START_FEN}|tactics=${state.analysisSettings.tacticsEnabled ? 1 : 0}|${history.map((move) => (move && move.san ? move.san : '')).join(' ')}`;
        let hash = 2166136261;
        for (let i = 0; i < seed.length; i += 1) {
            hash ^= seed.charCodeAt(i);
            hash = Math.imul(hash, 16777619);
        }
        return `g${(hash >>> 0).toString(16)}`;
    }

    function clampInt(value, min, max, fallback) {
        if (!Number.isFinite(value)) {
            return fallback;
        }
        const rounded = Math.round(value);
        return Math.max(min, Math.min(max, rounded));
    }
})();
