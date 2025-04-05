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
	.ascii "1-2-1\0"
	.align	3
lC4:
	.ascii "1-2-2\0"
	.align	3
lC5:
	.ascii "1-2-3\0"
	.align	3
lC6:
	.ascii "1-3-1\0"
	.align	3
lC7:
	.ascii "1-3-2\0"
	.align	3
lC8:
	.ascii "1-3-3\0"
	.align	3
lC9:
	.ascii "1-4-1\0"
	.align	3
lC10:
	.ascii "1-4-2\0"
	.align	3
lC11:
	.ascii "1-4-3\0"
	.align	3
lC12:
	.ascii "1-5-1\0"
	.align	3
lC13:
	.ascii "1-5-2\0"
	.align	3
lC14:
	.ascii "1-5-3\0"
	.align	3
lC15:
	.ascii "2-1-1\0"
	.align	3
lC16:
	.ascii "2-1-2\0"
	.align	3
lC17:
	.ascii "2-1-3\0"
	.align	3
lC18:
	.ascii "2-2-1\0"
	.align	3
lC19:
	.ascii "2-2-2\0"
	.align	3
lC20:
	.ascii "2-2-3\0"
	.align	3
lC21:
	.ascii "2-3-1\0"
	.align	3
lC22:
	.ascii "2-3-2\0"
	.align	3
lC23:
	.ascii "2-3-3\0"
	.align	3
lC24:
	.ascii "2-4-1\0"
	.align	3
lC25:
	.ascii "2-4-2\0"
	.align	3
lC26:
	.ascii "2-4-3\0"
	.align	3
lC27:
	.ascii "2-5-1\0"
	.align	3
lC28:
	.ascii "2-5-2\0"
	.align	3
lC29:
	.ascii "2-5-3\0"
	.align	3
lC30:
	.ascii "3-1-1\0"
	.align	3
lC31:
	.ascii "3-1-2\0"
	.align	3
lC32:
	.ascii "3-1-3\0"
	.align	3
lC33:
	.ascii "3-2-1\0"
	.align	3
lC34:
	.ascii "3-2-2\0"
	.align	3
lC35:
	.ascii "3-2-3\0"
	.align	3
lC36:
	.ascii "3-3-1\0"
	.align	3
lC37:
	.ascii "3-3-2\0"
	.align	3
lC38:
	.ascii "3-3-3\0"
	.align	3
lC39:
	.ascii "3-4-1\0"
	.align	3
lC40:
	.ascii "3-4-2\0"
	.align	3
lC41:
	.ascii "3-4-3\0"
	.align	3
lC42:
	.ascii "3-5-1\0"
	.align	3
lC43:
	.ascii "3-5-2\0"
	.align	3
lC44:
	.ascii "3-5-3\0"
	.align	3
lC45:
	.ascii "4-1-1\0"
	.align	3
lC46:
	.ascii "4-1-2\0"
	.align	3
lC47:
	.ascii "4-1-3\0"
	.align	3
lC48:
	.ascii "4-2-1\0"
	.align	3
lC49:
	.ascii "4-2-2\0"
	.align	3
lC50:
	.ascii "4-2-3\0"
	.align	3
lC51:
	.ascii "4-3-1\0"
	.align	3
lC52:
	.ascii "4-3-2\0"
	.align	3
lC53:
	.ascii "4-3-3\0"
	.align	3
lC54:
	.ascii "4-4-1\0"
	.align	3
lC55:
	.ascii "4-4-2\0"
	.align	3
lC56:
	.ascii "4-4-3\0"
	.align	3
lC57:
	.ascii "4-5-1\0"
	.align	3
lC58:
	.ascii "4-5-2\0"
	.align	3
lC59:
	.ascii "4-5-3\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 4
	str	w0, [x29, 28]
	mov	w0, 5
	str	w0, [x29, 24]
	mov	w0, 3
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 4
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 4
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 3
	beq	L4
	ldr	w0, [x29, 28]
	cmp	w0, 3
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L5
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L6
	b	L3
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L7
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L112
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L9
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L112
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L10
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L112
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L12
	b	L112
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L13
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L113
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L15
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L16
	b	L113
L15:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L14
L16:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	b	L14
L13:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	nop
L14:
	b	L113
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L17
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L114
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L19
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L20
	b	L114
L19:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	b	L18
L20:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L18
L17:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
L18:
	b	L114
L10:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L21
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L115
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L23
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L24
	b	L115
L23:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L22
L24:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	b	L22
L21:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	nop
L22:
	b	L115
L9:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L25
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L116
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L27
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L28
	b	L116
L27:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	b	L26
L28:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L26
L25:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L26:
	b	L116
L7:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L29
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L117
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L31
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L32
	b	L117
L31:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L30
L32:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	b	L30
L29:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	nop
L30:
	b	L117
L113:
	nop
	b	L112
L114:
	nop
	b	L112
L115:
	nop
	b	L112
L116:
	nop
	b	L112
L117:
	nop
	b	L112
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L33
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L118
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L35
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L118
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L36
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L118
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L37
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L38
	b	L118
L37:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L39
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L119
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L41
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L42
	b	L119
L41:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	b	L40
L42:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L40
L39:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L40:
	b	L119
L38:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L43
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L120
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L45
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L46
	b	L120
L45:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L44
L46:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	b	L44
L43:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	nop
L44:
	b	L120
L36:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L47
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L121
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L49
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L50
	b	L121
L49:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	b	L48
L50:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L48
L47:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L48:
	b	L121
L35:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L51
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L122
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L53
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L54
	b	L122
L53:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L52
L54:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	b	L52
L51:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	nop
L52:
	b	L122
L33:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L55
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L123
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L57
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L58
	b	L123
L57:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	b	L56
L58:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L56
L55:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
L56:
	b	L123
L119:
	nop
	b	L118
L120:
	nop
	b	L118
L121:
	nop
	b	L118
L122:
	nop
	b	L118
L123:
	nop
	b	L118
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L59
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L124
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L61
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L124
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L62
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L124
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L63
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L64
	b	L124
L63:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L65
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L125
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L67
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L68
	b	L125
L67:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L66
L68:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	b	L66
L65:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	nop
L66:
	b	L125
L64:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L69
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L126
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L71
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L72
	b	L126
L71:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	b	L70
L72:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L70
L69:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L70:
	b	L126
L62:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L73
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L127
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L75
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L76
	b	L127
L75:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L74
L76:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	b	L74
L73:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	nop
L74:
	b	L127
L61:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L77
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L128
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L79
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L80
	b	L128
L79:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	b	L78
L80:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L78
L77:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
L78:
	b	L128
L59:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L81
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L129
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L83
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L84
	b	L129
L83:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L82
L84:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	b	L82
L81:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	nop
L82:
	b	L129
L125:
	nop
	b	L124
L126:
	nop
	b	L124
L127:
	nop
	b	L124
L128:
	nop
	b	L124
L129:
	nop
	b	L124
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L85
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L130
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L87
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L130
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L88
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L130
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L89
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L90
	b	L130
L89:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L91
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L131
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L93
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L94
	b	L131
L93:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	b	L92
L94:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L92
L91:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
L92:
	b	L131
L90:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L95
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L132
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L97
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L98
	b	L132
L97:
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	b	L96
L98:
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	b	L96
L95:
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	nop
L96:
	b	L132
L88:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L99
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L133
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L101
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L102
	b	L133
L101:
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	b	L100
L102:
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	b	L100
L99:
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	nop
L100:
	b	L133
L87:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L103
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L134
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L105
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L106
	b	L134
L105:
	adrp	x0, lC54@PAGE
	add	x0, x0, lC54@PAGEOFF;
	bl	_puts
	b	L104
L106:
	adrp	x0, lC55@PAGE
	add	x0, x0, lC55@PAGEOFF;
	bl	_puts
	b	L104
L103:
	adrp	x0, lC56@PAGE
	add	x0, x0, lC56@PAGEOFF;
	bl	_puts
	nop
L104:
	b	L134
L85:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L107
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L135
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L109
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L110
	b	L135
L109:
	adrp	x0, lC57@PAGE
	add	x0, x0, lC57@PAGEOFF;
	bl	_puts
	b	L108
L110:
	adrp	x0, lC58@PAGE
	add	x0, x0, lC58@PAGEOFF;
	bl	_puts
	b	L108
L107:
	adrp	x0, lC59@PAGE
	add	x0, x0, lC59@PAGEOFF;
	bl	_puts
	nop
L108:
	b	L135
L131:
	nop
	b	L130
L132:
	nop
	b	L130
L133:
	nop
	b	L130
L134:
	nop
	b	L130
L135:
	nop
	b	L130
L112:
	nop
	b	L3
L118:
	nop
	b	L3
L124:
	nop
	b	L3
L130:
	nop
L3:
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
