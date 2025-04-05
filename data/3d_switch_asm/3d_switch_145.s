	.arch armv8.4-a+fp16+sb+ssbs
	.build_version macos,  15, 0
	.text
	.cstring
	.align	3
lC0:
	.ascii "1-1-1\0"
	.align	3
lC1:
	.ascii "1-1-2\0"
	.align	3
lC2:
	.ascii "1-1-3\0"
	.align	3
lC3:
	.ascii "1-1-4\0"
	.align	3
lC4:
	.ascii "1-1-5\0"
	.align	3
lC5:
	.ascii "1-2-1\0"
	.align	3
lC6:
	.ascii "1-2-2\0"
	.align	3
lC7:
	.ascii "1-2-3\0"
	.align	3
lC8:
	.ascii "1-2-4\0"
	.align	3
lC9:
	.ascii "1-2-5\0"
	.align	3
lC10:
	.ascii "1-3-1\0"
	.align	3
lC11:
	.ascii "1-3-2\0"
	.align	3
lC12:
	.ascii "1-3-3\0"
	.align	3
lC13:
	.ascii "1-3-4\0"
	.align	3
lC14:
	.ascii "1-3-5\0"
	.align	3
lC15:
	.ascii "1-4-1\0"
	.align	3
lC16:
	.ascii "1-4-2\0"
	.align	3
lC17:
	.ascii "1-4-3\0"
	.align	3
lC18:
	.ascii "1-4-4\0"
	.align	3
lC19:
	.ascii "1-4-5\0"
	.align	3
lC20:
	.ascii "1-5-1\0"
	.align	3
lC21:
	.ascii "1-5-2\0"
	.align	3
lC22:
	.ascii "1-5-3\0"
	.align	3
lC23:
	.ascii "1-5-4\0"
	.align	3
lC24:
	.ascii "1-5-5\0"
	.align	3
lC25:
	.ascii "2-1-1\0"
	.align	3
lC26:
	.ascii "2-1-2\0"
	.align	3
lC27:
	.ascii "2-1-3\0"
	.align	3
lC28:
	.ascii "2-1-4\0"
	.align	3
lC29:
	.ascii "2-1-5\0"
	.align	3
lC30:
	.ascii "2-2-1\0"
	.align	3
lC31:
	.ascii "2-2-2\0"
	.align	3
lC32:
	.ascii "2-2-3\0"
	.align	3
lC33:
	.ascii "2-2-4\0"
	.align	3
lC34:
	.ascii "2-2-5\0"
	.align	3
lC35:
	.ascii "2-3-1\0"
	.align	3
lC36:
	.ascii "2-3-2\0"
	.align	3
lC37:
	.ascii "2-3-3\0"
	.align	3
lC38:
	.ascii "2-3-4\0"
	.align	3
lC39:
	.ascii "2-3-5\0"
	.align	3
lC40:
	.ascii "2-4-1\0"
	.align	3
lC41:
	.ascii "2-4-2\0"
	.align	3
lC42:
	.ascii "2-4-3\0"
	.align	3
lC43:
	.ascii "2-4-4\0"
	.align	3
lC44:
	.ascii "2-4-5\0"
	.align	3
lC45:
	.ascii "2-5-1\0"
	.align	3
lC46:
	.ascii "2-5-2\0"
	.align	3
lC47:
	.ascii "2-5-3\0"
	.align	3
lC48:
	.ascii "2-5-4\0"
	.align	3
lC49:
	.ascii "2-5-5\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 2
	str	w0, [x29, 28]
	mov	w0, 5
	str	w0, [x29, 24]
	mov	w0, 5
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L3
	b	L4
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L5
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L78
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L7
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L78
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L8
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L78
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L9
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L10
	b	L78
L9:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L11
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L79
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L13
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L79
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L14
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L79
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L15
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L16
	b	L79
L15:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L12
L16:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	b	L12
L14:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L12
L13:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	b	L12
L11:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	nop
L12:
	b	L79
L10:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L17
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L80
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L19
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L80
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L20
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L80
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L21
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L22
	b	L80
L21:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	b	L18
L22:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L18
L20:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	b	L18
L19:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L18
L17:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
L18:
	b	L80
L8:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L23
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L81
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L25
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L81
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L26
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L81
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L27
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L28
	b	L81
L27:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L24
L28:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	b	L24
L26:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L24
L25:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	b	L24
L23:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	nop
L24:
	b	L81
L7:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L29
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L82
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L31
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L82
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L32
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L82
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L33
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L34
	b	L82
L33:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	b	L30
L34:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L30
L32:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	b	L30
L31:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L30
L29:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
L30:
	b	L82
L5:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L35
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L83
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L37
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L83
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L38
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L83
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L39
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L40
	b	L83
L39:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L36
L40:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	b	L36
L38:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L36
L37:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	b	L36
L35:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	nop
L36:
	b	L83
L79:
	nop
	b	L78
L80:
	nop
	b	L78
L81:
	nop
	b	L78
L82:
	nop
	b	L78
L83:
	nop
	b	L78
L3:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L41
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L84
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L43
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L84
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L44
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L84
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L45
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L46
	b	L84
L45:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L47
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L85
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L49
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L85
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L50
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L85
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L51
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L52
	b	L85
L51:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	b	L48
L52:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	b	L48
L50:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	b	L48
L49:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L48
L47:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
L48:
	b	L85
L46:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L53
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L86
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L55
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L86
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L56
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L86
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L57
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L58
	b	L86
L57:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L54
L58:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	b	L54
L56:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L54
L55:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	b	L54
L53:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	nop
L54:
	b	L86
L44:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L59
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L87
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L61
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L87
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L62
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L87
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L63
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L64
	b	L87
L63:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	b	L60
L64:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L60
L62:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	b	L60
L61:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L60
L59:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
L60:
	b	L87
L43:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L65
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L88
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L67
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L88
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L68
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L88
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L69
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L70
	b	L88
L69:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L66
L70:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	b	L66
L68:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L66
L67:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	b	L66
L65:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	nop
L66:
	b	L88
L41:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L71
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L89
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L73
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L89
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L74
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L89
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L75
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L76
	b	L89
L75:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	b	L72
L76:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L72
L74:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	b	L72
L73:
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	b	L72
L71:
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	nop
L72:
	b	L89
L85:
	nop
	b	L84
L86:
	nop
	b	L84
L87:
	nop
	b	L84
L88:
	nop
	b	L84
L89:
	nop
	b	L84
L78:
	nop
	b	L4
L84:
	nop
L4:
	mov	w0, 0
	ldp	x29, x30, [sp], 32
LCFI2:
	ret
LFE1:
	.section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
EH_frame1:
	.set L$set$0,LECIE1-LSCIE1
	.long L$set$0
LSCIE1:
	.long	0
	.byte	0x3
	.ascii "zR\0"
	.uleb128 0x1
	.sleb128 -8
	.uleb128 0x1e
	.uleb128 0x1
	.byte	0x10
	.byte	0xc
	.uleb128 0x1f
	.uleb128 0
	.align	3
LECIE1:
LSFDE1:
	.set L$set$1,LEFDE1-LASFDE1
	.long L$set$1
LASFDE1:
	.long	LASFDE1-EH_frame1
	.quad	LFB1-.
	.set L$set$2,LFE1-LFB1
	.quad L$set$2
	.uleb128 0
	.byte	0x4
	.set L$set$3,LCFI0-LFB1
	.long L$set$3
	.byte	0xe
	.uleb128 0x20
	.byte	0x9d
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x3
	.byte	0x4
	.set L$set$4,LCFI1-LCFI0
	.long L$set$4
	.byte	0xd
	.uleb128 0x1d
	.byte	0x4
	.set L$set$5,LCFI2-LCFI1
	.long L$set$5
	.byte	0xde
	.byte	0xdd
	.byte	0xc
	.uleb128 0x1f
	.uleb128 0
	.align	3
LEFDE1:
	.ident	"GCC: (Homebrew GCC 14.2.0_1) 14.2.0"
	.subsections_via_symbols
