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
	.ascii "2-1-1\0"
	.align	3
lC10:
	.ascii "2-1-2\0"
	.align	3
lC11:
	.ascii "2-1-3\0"
	.align	3
lC12:
	.ascii "2-2-1\0"
	.align	3
lC13:
	.ascii "2-2-2\0"
	.align	3
lC14:
	.ascii "2-2-3\0"
	.align	3
lC15:
	.ascii "2-3-1\0"
	.align	3
lC16:
	.ascii "2-3-2\0"
	.align	3
lC17:
	.ascii "2-3-3\0"
	.align	3
lC18:
	.ascii "3-1-1\0"
	.align	3
lC19:
	.ascii "3-1-2\0"
	.align	3
lC20:
	.ascii "3-1-3\0"
	.align	3
lC21:
	.ascii "3-2-1\0"
	.align	3
lC22:
	.ascii "3-2-2\0"
	.align	3
lC23:
	.ascii "3-2-3\0"
	.align	3
lC24:
	.ascii "3-3-1\0"
	.align	3
lC25:
	.ascii "3-3-2\0"
	.align	3
lC26:
	.ascii "3-3-3\0"
	.align	3
lC27:
	.ascii "4-1-1\0"
	.align	3
lC28:
	.ascii "4-1-2\0"
	.align	3
lC29:
	.ascii "4-1-3\0"
	.align	3
lC30:
	.ascii "4-2-1\0"
	.align	3
lC31:
	.ascii "4-2-2\0"
	.align	3
lC32:
	.ascii "4-2-3\0"
	.align	3
lC33:
	.ascii "4-3-1\0"
	.align	3
lC34:
	.ascii "4-3-2\0"
	.align	3
lC35:
	.ascii "4-3-3\0"
	.align	3
lC36:
	.ascii "5-1-1\0"
	.align	3
lC37:
	.ascii "5-1-2\0"
	.align	3
lC38:
	.ascii "5-1-3\0"
	.align	3
lC39:
	.ascii "5-2-1\0"
	.align	3
lC40:
	.ascii "5-2-2\0"
	.align	3
lC41:
	.ascii "5-2-3\0"
	.align	3
lC42:
	.ascii "5-3-1\0"
	.align	3
lC43:
	.ascii "5-3-2\0"
	.align	3
lC44:
	.ascii "5-3-3\0"
	.align	3
lC45:
	.ascii "6-1-1\0"
	.align	3
lC46:
	.ascii "6-1-2\0"
	.align	3
lC47:
	.ascii "6-1-3\0"
	.align	3
lC48:
	.ascii "6-2-1\0"
	.align	3
lC49:
	.ascii "6-2-2\0"
	.align	3
lC50:
	.ascii "6-2-3\0"
	.align	3
lC51:
	.ascii "6-3-1\0"
	.align	3
lC52:
	.ascii "6-3-2\0"
	.align	3
lC53:
	.ascii "6-3-3\0"
	.align	3
lC54:
	.ascii "7-1-1\0"
	.align	3
lC55:
	.ascii "7-1-2\0"
	.align	3
lC56:
	.ascii "7-1-3\0"
	.align	3
lC57:
	.ascii "7-2-1\0"
	.align	3
lC58:
	.ascii "7-2-2\0"
	.align	3
lC59:
	.ascii "7-2-3\0"
	.align	3
lC60:
	.ascii "7-3-1\0"
	.align	3
lC61:
	.ascii "7-3-2\0"
	.align	3
lC62:
	.ascii "7-3-3\0"
	.align	3
lC63:
	.ascii "8-1-1\0"
	.align	3
lC64:
	.ascii "8-1-2\0"
	.align	3
lC65:
	.ascii "8-1-3\0"
	.align	3
lC66:
	.ascii "8-2-1\0"
	.align	3
lC67:
	.ascii "8-2-2\0"
	.align	3
lC68:
	.ascii "8-2-3\0"
	.align	3
lC69:
	.ascii "8-3-1\0"
	.align	3
lC70:
	.ascii "8-3-2\0"
	.align	3
lC71:
	.ascii "8-3-3\0"
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
	mov	w0, 3
	str	w0, [x29, 24]
	mov	w0, 3
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
	cmp	w0, 3
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L140
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L14
	b	L140
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L15
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L141
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L17
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L18
	b	L141
L17:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L16
L18:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	b	L16
L15:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	nop
L16:
	b	L141
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L19
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L142
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L21
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L22
	b	L142
L21:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	b	L20
L22:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L20
L19:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
L20:
	b	L142
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L23
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L143
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L25
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L26
	b	L143
L25:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L24
L26:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	b	L24
L23:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	nop
L24:
	b	L143
L141:
	nop
	b	L140
L142:
	nop
	b	L140
L143:
	nop
	b	L140
L10:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L27
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L144
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L29
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L30
	b	L144
L29:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L31
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L145
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L33
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L34
	b	L145
L33:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	b	L32
L34:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L32
L31:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L32:
	b	L145
L30:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L35
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L146
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L37
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L38
	b	L146
L37:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L36
L38:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	b	L36
L35:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	nop
L36:
	b	L146
L27:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L39
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L147
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L41
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L42
	b	L147
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
	b	L147
L145:
	nop
	b	L144
L146:
	nop
	b	L144
L147:
	nop
	b	L144
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L43
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L148
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L45
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L46
	b	L148
L45:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L47
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L149
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L49
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L50
	b	L149
L49:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L48
L50:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	b	L48
L47:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	nop
L48:
	b	L149
L46:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L51
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L150
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L53
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L54
	b	L150
L53:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	b	L52
L54:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L52
L51:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L52:
	b	L150
L43:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L55
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L151
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L57
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L58
	b	L151
L57:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L56
L58:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	b	L56
L55:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	nop
L56:
	b	L151
L149:
	nop
	b	L148
L150:
	nop
	b	L148
L151:
	nop
	b	L148
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L59
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L152
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L61
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L62
	b	L152
L61:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L63
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L153
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L65
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L66
	b	L153
L65:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	b	L64
L66:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L64
L63:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
L64:
	b	L153
L62:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L67
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L154
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L69
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L70
	b	L154
L69:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L68
L70:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	b	L68
L67:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	nop
L68:
	b	L154
L59:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L71
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L155
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L73
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L74
	b	L155
L73:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	b	L72
L74:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L72
L71:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L72:
	b	L155
L153:
	nop
	b	L152
L154:
	nop
	b	L152
L155:
	nop
	b	L152
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L75
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L156
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L77
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L78
	b	L156
L77:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L79
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L157
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L81
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L82
	b	L157
L81:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L80
L82:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	b	L80
L79:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	nop
L80:
	b	L157
L78:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L83
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L158
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L85
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L86
	b	L158
L85:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	b	L84
L86:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L84
L83:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
L84:
	b	L158
L75:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L87
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L159
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L89
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L90
	b	L159
L89:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L88
L90:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	b	L88
L87:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	nop
L88:
	b	L159
L157:
	nop
	b	L156
L158:
	nop
	b	L156
L159:
	nop
	b	L156
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L91
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L160
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L93
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L94
	b	L160
L93:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L95
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L161
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L97
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L98
	b	L161
L97:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	b	L96
L98:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L96
L95:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
L96:
	b	L161
L94:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L99
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L162
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L101
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L102
	b	L162
L101:
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	b	L100
L102:
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	b	L100
L99:
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	nop
L100:
	b	L162
L91:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L103
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L163
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L105
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L106
	b	L163
L105:
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	b	L104
L106:
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	b	L104
L103:
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	nop
L104:
	b	L163
L161:
	nop
	b	L160
L162:
	nop
	b	L160
L163:
	nop
	b	L160
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L107
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L164
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L109
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L110
	b	L164
L109:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L111
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L165
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L113
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L114
	b	L165
L113:
	adrp	x0, lC54@PAGE
	add	x0, x0, lC54@PAGEOFF;
	bl	_puts
	b	L112
L114:
	adrp	x0, lC55@PAGE
	add	x0, x0, lC55@PAGEOFF;
	bl	_puts
	b	L112
L111:
	adrp	x0, lC56@PAGE
	add	x0, x0, lC56@PAGEOFF;
	bl	_puts
	nop
L112:
	b	L165
L110:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L115
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L166
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L117
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L118
	b	L166
L117:
	adrp	x0, lC57@PAGE
	add	x0, x0, lC57@PAGEOFF;
	bl	_puts
	b	L116
L118:
	adrp	x0, lC58@PAGE
	add	x0, x0, lC58@PAGEOFF;
	bl	_puts
	b	L116
L115:
	adrp	x0, lC59@PAGE
	add	x0, x0, lC59@PAGEOFF;
	bl	_puts
	nop
L116:
	b	L166
L107:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L119
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L167
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L121
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L122
	b	L167
L121:
	adrp	x0, lC60@PAGE
	add	x0, x0, lC60@PAGEOFF;
	bl	_puts
	b	L120
L122:
	adrp	x0, lC61@PAGE
	add	x0, x0, lC61@PAGEOFF;
	bl	_puts
	b	L120
L119:
	adrp	x0, lC62@PAGE
	add	x0, x0, lC62@PAGEOFF;
	bl	_puts
	nop
L120:
	b	L167
L165:
	nop
	b	L164
L166:
	nop
	b	L164
L167:
	nop
	b	L164
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L123
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L168
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L125
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L126
	b	L168
L125:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L127
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L169
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L129
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L130
	b	L169
L129:
	adrp	x0, lC63@PAGE
	add	x0, x0, lC63@PAGEOFF;
	bl	_puts
	b	L128
L130:
	adrp	x0, lC64@PAGE
	add	x0, x0, lC64@PAGEOFF;
	bl	_puts
	b	L128
L127:
	adrp	x0, lC65@PAGE
	add	x0, x0, lC65@PAGEOFF;
	bl	_puts
	nop
L128:
	b	L169
L126:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L131
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L170
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L133
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L134
	b	L170
L133:
	adrp	x0, lC66@PAGE
	add	x0, x0, lC66@PAGEOFF;
	bl	_puts
	b	L132
L134:
	adrp	x0, lC67@PAGE
	add	x0, x0, lC67@PAGEOFF;
	bl	_puts
	b	L132
L131:
	adrp	x0, lC68@PAGE
	add	x0, x0, lC68@PAGEOFF;
	bl	_puts
	nop
L132:
	b	L170
L123:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L135
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L171
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L137
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L138
	b	L171
L137:
	adrp	x0, lC69@PAGE
	add	x0, x0, lC69@PAGEOFF;
	bl	_puts
	b	L136
L138:
	adrp	x0, lC70@PAGE
	add	x0, x0, lC70@PAGEOFF;
	bl	_puts
	b	L136
L135:
	adrp	x0, lC71@PAGE
	add	x0, x0, lC71@PAGEOFF;
	bl	_puts
	nop
L136:
	b	L171
L169:
	nop
	b	L168
L170:
	nop
	b	L168
L171:
	nop
	b	L168
L140:
	nop
	b	L3
L144:
	nop
	b	L3
L148:
	nop
	b	L3
L152:
	nop
	b	L3
L156:
	nop
	b	L3
L160:
	nop
	b	L3
L164:
	nop
	b	L3
L168:
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
