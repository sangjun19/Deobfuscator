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
	.ascii "1-2-1\0"
	.align	3
lC5:
	.ascii "1-2-2\0"
	.align	3
lC6:
	.ascii "1-2-3\0"
	.align	3
lC7:
	.ascii "1-2-4\0"
	.align	3
lC8:
	.ascii "2-1-1\0"
	.align	3
lC9:
	.ascii "2-1-2\0"
	.align	3
lC10:
	.ascii "2-1-3\0"
	.align	3
lC11:
	.ascii "2-1-4\0"
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
	.ascii "2-2-4\0"
	.align	3
lC16:
	.ascii "3-1-1\0"
	.align	3
lC17:
	.ascii "3-1-2\0"
	.align	3
lC18:
	.ascii "3-1-3\0"
	.align	3
lC19:
	.ascii "3-1-4\0"
	.align	3
lC20:
	.ascii "3-2-1\0"
	.align	3
lC21:
	.ascii "3-2-2\0"
	.align	3
lC22:
	.ascii "3-2-3\0"
	.align	3
lC23:
	.ascii "3-2-4\0"
	.align	3
lC24:
	.ascii "4-1-1\0"
	.align	3
lC25:
	.ascii "4-1-2\0"
	.align	3
lC26:
	.ascii "4-1-3\0"
	.align	3
lC27:
	.ascii "4-1-4\0"
	.align	3
lC28:
	.ascii "4-2-1\0"
	.align	3
lC29:
	.ascii "4-2-2\0"
	.align	3
lC30:
	.ascii "4-2-3\0"
	.align	3
lC31:
	.ascii "4-2-4\0"
	.align	3
lC32:
	.ascii "5-1-1\0"
	.align	3
lC33:
	.ascii "5-1-2\0"
	.align	3
lC34:
	.ascii "5-1-3\0"
	.align	3
lC35:
	.ascii "5-1-4\0"
	.align	3
lC36:
	.ascii "5-2-1\0"
	.align	3
lC37:
	.ascii "5-2-2\0"
	.align	3
lC38:
	.ascii "5-2-3\0"
	.align	3
lC39:
	.ascii "5-2-4\0"
	.align	3
lC40:
	.ascii "6-1-1\0"
	.align	3
lC41:
	.ascii "6-1-2\0"
	.align	3
lC42:
	.ascii "6-1-3\0"
	.align	3
lC43:
	.ascii "6-1-4\0"
	.align	3
lC44:
	.ascii "6-2-1\0"
	.align	3
lC45:
	.ascii "6-2-2\0"
	.align	3
lC46:
	.ascii "6-2-3\0"
	.align	3
lC47:
	.ascii "6-2-4\0"
	.align	3
lC48:
	.ascii "7-1-1\0"
	.align	3
lC49:
	.ascii "7-1-2\0"
	.align	3
lC50:
	.ascii "7-1-3\0"
	.align	3
lC51:
	.ascii "7-1-4\0"
	.align	3
lC52:
	.ascii "7-2-1\0"
	.align	3
lC53:
	.ascii "7-2-2\0"
	.align	3
lC54:
	.ascii "7-2-3\0"
	.align	3
lC55:
	.ascii "7-2-4\0"
	.align	3
lC56:
	.ascii "8-1-1\0"
	.align	3
lC57:
	.ascii "8-1-2\0"
	.align	3
lC58:
	.ascii "8-1-3\0"
	.align	3
lC59:
	.ascii "8-1-4\0"
	.align	3
lC60:
	.ascii "8-2-1\0"
	.align	3
lC61:
	.ascii "8-2-2\0"
	.align	3
lC62:
	.ascii "8-2-3\0"
	.align	3
lC63:
	.ascii "8-2-4\0"
	.align	3
lC64:
	.ascii "9-1-1\0"
	.align	3
lC65:
	.ascii "9-1-2\0"
	.align	3
lC66:
	.ascii "9-1-3\0"
	.align	3
lC67:
	.ascii "9-1-4\0"
	.align	3
lC68:
	.ascii "9-2-1\0"
	.align	3
lC69:
	.ascii "9-2-2\0"
	.align	3
lC70:
	.ascii "9-2-3\0"
	.align	3
lC71:
	.ascii "9-2-4\0"
	.align	3
lC72:
	.ascii "10-1-1\0"
	.align	3
lC73:
	.ascii "10-1-2\0"
	.align	3
lC74:
	.ascii "10-1-3\0"
	.align	3
lC75:
	.ascii "10-1-4\0"
	.align	3
lC76:
	.ascii "10-2-1\0"
	.align	3
lC77:
	.ascii "10-2-2\0"
	.align	3
lC78:
	.ascii "10-2-3\0"
	.align	3
lC79:
	.ascii "10-2-4\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 10
	str	w0, [x29, 28]
	mov	w0, 2
	str	w0, [x29, 24]
	mov	w0, 4
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 10
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 10
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 9
	beq	L4
	ldr	w0, [x29, 28]
	cmp	w0, 9
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 8
	beq	L5
	ldr	w0, [x29, 28]
	cmp	w0, 8
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 7
	beq	L6
	ldr	w0, [x29, 28]
	cmp	w0, 7
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 6
	beq	L7
	ldr	w0, [x29, 28]
	cmp	w0, 6
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 5
	beq	L8
	ldr	w0, [x29, 28]
	cmp	w0, 5
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 4
	beq	L9
	ldr	w0, [x29, 28]
	cmp	w0, 4
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 3
	beq	L10
	ldr	w0, [x29, 28]
	cmp	w0, 3
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L11
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L12
	b	L3
L11:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L14
	b	L3
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L16
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L144
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L18
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L144
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L19
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L20
	b	L144
L19:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L17
L20:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	b	L17
L18:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L17
L16:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
L17:
	b	L144
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L21
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L145
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L23
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L145
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L24
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L25
	b	L145
L24:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L22
L25:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	b	L22
L23:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L22
L21:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
L22:
	b	L145
L144:
	nop
	b	L3
L145:
	nop
	b	L3
L12:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L26
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L27
	b	L3
L26:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L29
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L146
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L31
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L146
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L32
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L33
	b	L146
L32:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L30
L33:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	b	L30
L31:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L30
L29:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L30:
	b	L146
L27:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L34
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L147
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L36
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L147
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L37
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L38
	b	L147
L37:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L35
L38:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	b	L35
L36:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L35
L34:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
L35:
	b	L147
L146:
	nop
	b	L3
L147:
	nop
	b	L3
L10:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L39
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L40
	b	L3
L39:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L42
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L148
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L44
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L148
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L45
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L46
	b	L148
L45:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L43
L46:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	b	L43
L44:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L43
L42:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
L43:
	b	L148
L40:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L47
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L149
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L49
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L149
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L50
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L51
	b	L149
L50:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L48
L51:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	b	L48
L49:
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
	b	L149
L148:
	nop
	b	L3
L149:
	nop
	b	L3
L9:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L52
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L53
	b	L3
L52:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L55
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L150
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L57
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L150
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L58
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L59
	b	L150
L58:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L56
L59:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	b	L56
L57:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	b	L56
L55:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
L56:
	b	L150
L53:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L60
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L151
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L62
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L151
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L63
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L64
	b	L151
L63:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L61
L64:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	b	L61
L62:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L61
L60:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
L61:
	b	L151
L150:
	nop
	b	L3
L151:
	nop
	b	L3
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L65
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L66
	b	L3
L65:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L68
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L152
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L70
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L152
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L71
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L72
	b	L152
L71:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L69
L72:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	b	L69
L70:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L69
L68:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L69:
	b	L152
L66:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L73
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L153
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L75
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L153
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L76
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L77
	b	L153
L76:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L74
L77:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	b	L74
L75:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L74
L73:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
L74:
	b	L153
L152:
	nop
	b	L3
L153:
	nop
	b	L3
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L78
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L79
	b	L3
L78:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L81
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L154
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L83
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L154
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L84
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L85
	b	L154
L84:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L82
L85:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	b	L82
L83:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L82
L81:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	nop
L82:
	b	L154
L79:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L86
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L155
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L88
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L155
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L89
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L90
	b	L155
L89:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	b	L87
L90:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	b	L87
L88:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L87
L86:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
L87:
	b	L155
L154:
	nop
	b	L3
L155:
	nop
	b	L3
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L91
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L92
	b	L3
L91:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L94
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L156
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L96
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L156
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L97
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L98
	b	L156
L97:
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	b	L95
L98:
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	b	L95
L96:
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	b	L95
L94:
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	nop
L95:
	b	L156
L92:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L99
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L157
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L101
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L157
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L102
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L103
	b	L157
L102:
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	b	L100
L103:
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	b	L100
L101:
	adrp	x0, lC54@PAGE
	add	x0, x0, lC54@PAGEOFF;
	bl	_puts
	b	L100
L99:
	adrp	x0, lC55@PAGE
	add	x0, x0, lC55@PAGEOFF;
	bl	_puts
	nop
L100:
	b	L157
L156:
	nop
	b	L3
L157:
	nop
	b	L3
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L104
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L105
	b	L3
L104:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L107
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L158
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L109
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L158
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L110
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L111
	b	L158
L110:
	adrp	x0, lC56@PAGE
	add	x0, x0, lC56@PAGEOFF;
	bl	_puts
	b	L108
L111:
	adrp	x0, lC57@PAGE
	add	x0, x0, lC57@PAGEOFF;
	bl	_puts
	b	L108
L109:
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
	b	L158
L105:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L112
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L159
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L114
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L159
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L115
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L116
	b	L159
L115:
	adrp	x0, lC60@PAGE
	add	x0, x0, lC60@PAGEOFF;
	bl	_puts
	b	L113
L116:
	adrp	x0, lC61@PAGE
	add	x0, x0, lC61@PAGEOFF;
	bl	_puts
	b	L113
L114:
	adrp	x0, lC62@PAGE
	add	x0, x0, lC62@PAGEOFF;
	bl	_puts
	b	L113
L112:
	adrp	x0, lC63@PAGE
	add	x0, x0, lC63@PAGEOFF;
	bl	_puts
	nop
L113:
	b	L159
L158:
	nop
	b	L3
L159:
	nop
	b	L3
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L117
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L118
	b	L3
L117:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L120
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L160
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L122
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L160
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L123
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L124
	b	L160
L123:
	adrp	x0, lC64@PAGE
	add	x0, x0, lC64@PAGEOFF;
	bl	_puts
	b	L121
L124:
	adrp	x0, lC65@PAGE
	add	x0, x0, lC65@PAGEOFF;
	bl	_puts
	b	L121
L122:
	adrp	x0, lC66@PAGE
	add	x0, x0, lC66@PAGEOFF;
	bl	_puts
	b	L121
L120:
	adrp	x0, lC67@PAGE
	add	x0, x0, lC67@PAGEOFF;
	bl	_puts
	nop
L121:
	b	L160
L118:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L125
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L161
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L127
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L161
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L128
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L129
	b	L161
L128:
	adrp	x0, lC68@PAGE
	add	x0, x0, lC68@PAGEOFF;
	bl	_puts
	b	L126
L129:
	adrp	x0, lC69@PAGE
	add	x0, x0, lC69@PAGEOFF;
	bl	_puts
	b	L126
L127:
	adrp	x0, lC70@PAGE
	add	x0, x0, lC70@PAGEOFF;
	bl	_puts
	b	L126
L125:
	adrp	x0, lC71@PAGE
	add	x0, x0, lC71@PAGEOFF;
	bl	_puts
	nop
L126:
	b	L161
L160:
	nop
	b	L3
L161:
	nop
	b	L3
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L130
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L131
	b	L164
L130:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L133
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L162
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L135
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L162
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L136
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L137
	b	L162
L136:
	adrp	x0, lC72@PAGE
	add	x0, x0, lC72@PAGEOFF;
	bl	_puts
	b	L134
L137:
	adrp	x0, lC73@PAGE
	add	x0, x0, lC73@PAGEOFF;
	bl	_puts
	b	L134
L135:
	adrp	x0, lC74@PAGE
	add	x0, x0, lC74@PAGEOFF;
	bl	_puts
	b	L134
L133:
	adrp	x0, lC75@PAGE
	add	x0, x0, lC75@PAGEOFF;
	bl	_puts
	nop
L134:
	b	L162
L131:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L138
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L163
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L140
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L163
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L141
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L142
	b	L163
L141:
	adrp	x0, lC76@PAGE
	add	x0, x0, lC76@PAGEOFF;
	bl	_puts
	b	L139
L142:
	adrp	x0, lC77@PAGE
	add	x0, x0, lC77@PAGEOFF;
	bl	_puts
	b	L139
L140:
	adrp	x0, lC78@PAGE
	add	x0, x0, lC78@PAGEOFF;
	bl	_puts
	b	L139
L138:
	adrp	x0, lC79@PAGE
	add	x0, x0, lC79@PAGEOFF;
	bl	_puts
	nop
L139:
	b	L163
L162:
	nop
	b	L164
L163:
	nop
L164:
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
