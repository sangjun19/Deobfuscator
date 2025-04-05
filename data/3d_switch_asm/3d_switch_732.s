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
	.ascii "1-2-1\0"
	.align	3
lC3:
	.ascii "1-2-2\0"
	.align	3
lC4:
	.ascii "1-3-1\0"
	.align	3
lC5:
	.ascii "1-3-2\0"
	.align	3
lC6:
	.ascii "1-4-1\0"
	.align	3
lC7:
	.ascii "1-4-2\0"
	.align	3
lC8:
	.ascii "2-1-1\0"
	.align	3
lC9:
	.ascii "2-1-2\0"
	.align	3
lC10:
	.ascii "2-2-1\0"
	.align	3
lC11:
	.ascii "2-2-2\0"
	.align	3
lC12:
	.ascii "2-3-1\0"
	.align	3
lC13:
	.ascii "2-3-2\0"
	.align	3
lC14:
	.ascii "2-4-1\0"
	.align	3
lC15:
	.ascii "2-4-2\0"
	.align	3
lC16:
	.ascii "3-1-1\0"
	.align	3
lC17:
	.ascii "3-1-2\0"
	.align	3
lC18:
	.ascii "3-2-1\0"
	.align	3
lC19:
	.ascii "3-2-2\0"
	.align	3
lC20:
	.ascii "3-3-1\0"
	.align	3
lC21:
	.ascii "3-3-2\0"
	.align	3
lC22:
	.ascii "3-4-1\0"
	.align	3
lC23:
	.ascii "3-4-2\0"
	.align	3
lC24:
	.ascii "4-1-1\0"
	.align	3
lC25:
	.ascii "4-1-2\0"
	.align	3
lC26:
	.ascii "4-2-1\0"
	.align	3
lC27:
	.ascii "4-2-2\0"
	.align	3
lC28:
	.ascii "4-3-1\0"
	.align	3
lC29:
	.ascii "4-3-2\0"
	.align	3
lC30:
	.ascii "4-4-1\0"
	.align	3
lC31:
	.ascii "4-4-2\0"
	.align	3
lC32:
	.ascii "5-1-1\0"
	.align	3
lC33:
	.ascii "5-1-2\0"
	.align	3
lC34:
	.ascii "5-2-1\0"
	.align	3
lC35:
	.ascii "5-2-2\0"
	.align	3
lC36:
	.ascii "5-3-1\0"
	.align	3
lC37:
	.ascii "5-3-2\0"
	.align	3
lC38:
	.ascii "5-4-1\0"
	.align	3
lC39:
	.ascii "5-4-2\0"
	.align	3
lC40:
	.ascii "6-1-1\0"
	.align	3
lC41:
	.ascii "6-1-2\0"
	.align	3
lC42:
	.ascii "6-2-1\0"
	.align	3
lC43:
	.ascii "6-2-2\0"
	.align	3
lC44:
	.ascii "6-3-1\0"
	.align	3
lC45:
	.ascii "6-3-2\0"
	.align	3
lC46:
	.ascii "6-4-1\0"
	.align	3
lC47:
	.ascii "6-4-2\0"
	.align	3
lC48:
	.ascii "7-1-1\0"
	.align	3
lC49:
	.ascii "7-1-2\0"
	.align	3
lC50:
	.ascii "7-2-1\0"
	.align	3
lC51:
	.ascii "7-2-2\0"
	.align	3
lC52:
	.ascii "7-3-1\0"
	.align	3
lC53:
	.ascii "7-3-2\0"
	.align	3
lC54:
	.ascii "7-4-1\0"
	.align	3
lC55:
	.ascii "7-4-2\0"
	.align	3
lC56:
	.ascii "8-1-1\0"
	.align	3
lC57:
	.ascii "8-1-2\0"
	.align	3
lC58:
	.ascii "8-2-1\0"
	.align	3
lC59:
	.ascii "8-2-2\0"
	.align	3
lC60:
	.ascii "8-3-1\0"
	.align	3
lC61:
	.ascii "8-3-2\0"
	.align	3
lC62:
	.ascii "8-4-1\0"
	.align	3
lC63:
	.ascii "8-4-2\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 8
	str	w0, [x29, 28]
	mov	w0, 4
	str	w0, [x29, 24]
	mov	w0, 2
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 8
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 8
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 7
	beq	L4
	ldr	w0, [x29, 28]
	cmp	w0, 7
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 6
	beq	L5
	ldr	w0, [x29, 28]
	cmp	w0, 6
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 5
	beq	L6
	ldr	w0, [x29, 28]
	cmp	w0, 5
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 4
	beq	L7
	ldr	w0, [x29, 28]
	cmp	w0, 4
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 3
	beq	L8
	ldr	w0, [x29, 28]
	cmp	w0, 3
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L9
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L10
	b	L3
L9:
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L148
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L148
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L14
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L15
	b	L148
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L16
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L17
	b	L12
L16:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L18
L17:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	nop
L18:
	b	L12
L15:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L19
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L20
	b	L12
L19:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L21
L20:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
L21:
	b	L12
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L22
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L23
	b	L12
L22:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L24
L23:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
L24:
	b	L12
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L25
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L26
	b	L149
L25:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L27
L26:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
L27:
L149:
	nop
L12:
	b	L148
L10:
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L28
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L150
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L30
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L150
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L31
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L32
	b	L150
L31:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L33
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L34
	b	L29
L33:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L35
L34:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
L35:
	b	L29
L32:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L36
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L37
	b	L29
L36:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L38
L37:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L38:
	b	L29
L30:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L39
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L40
	b	L29
L39:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L41
L40:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
L41:
	b	L29
L28:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L42
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L43
	b	L151
L42:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L44
L43:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
L44:
L151:
	nop
L29:
	b	L150
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L45
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L152
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L47
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L152
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L48
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L49
	b	L152
L48:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L50
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L51
	b	L46
L50:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L52
L51:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L52:
	b	L46
L49:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L53
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L54
	b	L46
L53:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L55
L54:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
L55:
	b	L46
L47:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L56
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L57
	b	L46
L56:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L58
L57:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	nop
L58:
	b	L46
L45:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L59
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L60
	b	L153
L59:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L61
L60:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L61:
L153:
	nop
L46:
	b	L152
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L62
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L154
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L64
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L154
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L65
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L66
	b	L154
L65:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L67
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L68
	b	L63
L67:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L69
L68:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	nop
L69:
	b	L63
L66:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L70
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L71
	b	L63
L70:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	b	L72
L71:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
L72:
	b	L63
L64:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L73
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L74
	b	L63
L73:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L75
L74:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
L75:
	b	L63
L62:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L76
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L77
	b	L155
L76:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L78
L77:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
L78:
L155:
	nop
L63:
	b	L154
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L79
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L156
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L81
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L156
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L82
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L83
	b	L156
L82:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L84
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L85
	b	L80
L84:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L86
L85:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	nop
L86:
	b	L80
L83:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L87
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L88
	b	L80
L87:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L89
L88:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L89:
	b	L80
L81:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L90
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L91
	b	L80
L90:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L92
L91:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	nop
L92:
	b	L80
L79:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L93
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L94
	b	L157
L93:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L95
L94:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
L95:
L157:
	nop
L80:
	b	L156
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L96
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L158
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L98
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L158
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L99
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L100
	b	L158
L99:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L101
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L102
	b	L97
L101:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L103
L102:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
L103:
	b	L97
L100:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L104
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L105
	b	L97
L104:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L106
L105:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	nop
L106:
	b	L97
L98:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L107
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L108
	b	L97
L107:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	b	L109
L108:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	nop
L109:
	b	L97
L96:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L110
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L111
	b	L159
L110:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L112
L111:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
L112:
L159:
	nop
L97:
	b	L158
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L113
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L160
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L115
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L160
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L116
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L117
	b	L160
L116:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L118
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L119
	b	L114
L118:
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	b	L120
L119:
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	nop
L120:
	b	L114
L117:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L121
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L122
	b	L114
L121:
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	b	L123
L122:
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	nop
L123:
	b	L114
L115:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L124
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L125
	b	L114
L124:
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	b	L126
L125:
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	nop
L126:
	b	L114
L113:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L127
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L128
	b	L161
L127:
	adrp	x0, lC54@PAGE
	add	x0, x0, lC54@PAGEOFF;
	bl	_puts
	b	L129
L128:
	adrp	x0, lC55@PAGE
	add	x0, x0, lC55@PAGEOFF;
	bl	_puts
	nop
L129:
L161:
	nop
L114:
	b	L160
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L130
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L162
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L132
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L162
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L133
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L134
	b	L162
L133:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L135
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L136
	b	L131
L135:
	adrp	x0, lC56@PAGE
	add	x0, x0, lC56@PAGEOFF;
	bl	_puts
	b	L137
L136:
	adrp	x0, lC57@PAGE
	add	x0, x0, lC57@PAGEOFF;
	bl	_puts
	nop
L137:
	b	L131
L134:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L138
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L139
	b	L131
L138:
	adrp	x0, lC58@PAGE
	add	x0, x0, lC58@PAGEOFF;
	bl	_puts
	b	L140
L139:
	adrp	x0, lC59@PAGE
	add	x0, x0, lC59@PAGEOFF;
	bl	_puts
	nop
L140:
	b	L131
L132:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L141
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L142
	b	L131
L141:
	adrp	x0, lC60@PAGE
	add	x0, x0, lC60@PAGEOFF;
	bl	_puts
	b	L143
L142:
	adrp	x0, lC61@PAGE
	add	x0, x0, lC61@PAGEOFF;
	bl	_puts
	nop
L143:
	b	L131
L130:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L144
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L145
	b	L163
L144:
	adrp	x0, lC62@PAGE
	add	x0, x0, lC62@PAGEOFF;
	bl	_puts
	b	L146
L145:
	adrp	x0, lC63@PAGE
	add	x0, x0, lC63@PAGEOFF;
	bl	_puts
	nop
L146:
L163:
	nop
L131:
	b	L162
L148:
	nop
	b	L3
L150:
	nop
	b	L3
L152:
	nop
	b	L3
L154:
	nop
	b	L3
L156:
	nop
	b	L3
L158:
	nop
	b	L3
L160:
	nop
	b	L3
L162:
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
