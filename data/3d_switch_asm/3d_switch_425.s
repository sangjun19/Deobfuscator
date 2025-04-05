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
	.ascii "2-1-1\0"
	.align	3
lC16:
	.ascii "2-1-2\0"
	.align	3
lC17:
	.ascii "2-1-3\0"
	.align	3
lC18:
	.ascii "2-1-4\0"
	.align	3
lC19:
	.ascii "2-1-5\0"
	.align	3
lC20:
	.ascii "2-2-1\0"
	.align	3
lC21:
	.ascii "2-2-2\0"
	.align	3
lC22:
	.ascii "2-2-3\0"
	.align	3
lC23:
	.ascii "2-2-4\0"
	.align	3
lC24:
	.ascii "2-2-5\0"
	.align	3
lC25:
	.ascii "2-3-1\0"
	.align	3
lC26:
	.ascii "2-3-2\0"
	.align	3
lC27:
	.ascii "2-3-3\0"
	.align	3
lC28:
	.ascii "2-3-4\0"
	.align	3
lC29:
	.ascii "2-3-5\0"
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
	.ascii "3-1-4\0"
	.align	3
lC34:
	.ascii "3-1-5\0"
	.align	3
lC35:
	.ascii "3-2-1\0"
	.align	3
lC36:
	.ascii "3-2-2\0"
	.align	3
lC37:
	.ascii "3-2-3\0"
	.align	3
lC38:
	.ascii "3-2-4\0"
	.align	3
lC39:
	.ascii "3-2-5\0"
	.align	3
lC40:
	.ascii "3-3-1\0"
	.align	3
lC41:
	.ascii "3-3-2\0"
	.align	3
lC42:
	.ascii "3-3-3\0"
	.align	3
lC43:
	.ascii "3-3-4\0"
	.align	3
lC44:
	.ascii "3-3-5\0"
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
	.ascii "4-1-4\0"
	.align	3
lC49:
	.ascii "4-1-5\0"
	.align	3
lC50:
	.ascii "4-2-1\0"
	.align	3
lC51:
	.ascii "4-2-2\0"
	.align	3
lC52:
	.ascii "4-2-3\0"
	.align	3
lC53:
	.ascii "4-2-4\0"
	.align	3
lC54:
	.ascii "4-2-5\0"
	.align	3
lC55:
	.ascii "4-3-1\0"
	.align	3
lC56:
	.ascii "4-3-2\0"
	.align	3
lC57:
	.ascii "4-3-3\0"
	.align	3
lC58:
	.ascii "4-3-4\0"
	.align	3
lC59:
	.ascii "4-3-5\0"
	.align	3
lC60:
	.ascii "5-1-1\0"
	.align	3
lC61:
	.ascii "5-1-2\0"
	.align	3
lC62:
	.ascii "5-1-3\0"
	.align	3
lC63:
	.ascii "5-1-4\0"
	.align	3
lC64:
	.ascii "5-1-5\0"
	.align	3
lC65:
	.ascii "5-2-1\0"
	.align	3
lC66:
	.ascii "5-2-2\0"
	.align	3
lC67:
	.ascii "5-2-3\0"
	.align	3
lC68:
	.ascii "5-2-4\0"
	.align	3
lC69:
	.ascii "5-2-5\0"
	.align	3
lC70:
	.ascii "5-3-1\0"
	.align	3
lC71:
	.ascii "5-3-2\0"
	.align	3
lC72:
	.ascii "5-3-3\0"
	.align	3
lC73:
	.ascii "5-3-4\0"
	.align	3
lC74:
	.ascii "5-3-5\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 5
	str	w0, [x29, 28]
	mov	w0, 3
	str	w0, [x29, 24]
	mov	w0, 5
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 5
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 5
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 4
	beq	L4
	ldr	w0, [x29, 28]
	cmp	w0, 4
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 3
	beq	L5
	ldr	w0, [x29, 28]
	cmp	w0, 3
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L6
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L7
	b	L3
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L8
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L119
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L10
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L11
	b	L119
L10:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L12
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L120
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L14
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L120
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L15
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L120
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L16
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L17
	b	L120
L16:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L13
L17:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	b	L13
L15:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L13
L14:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	b	L13
L12:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	nop
L13:
	b	L120
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L18
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L121
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L20
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L121
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L21
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L121
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L22
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L23
	b	L121
L22:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	b	L19
L23:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L19
L21:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	b	L19
L20:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L19
L18:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
L19:
	b	L121
L8:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L24
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L122
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L26
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L122
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L27
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L122
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L28
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L29
	b	L122
L28:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L25
L29:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	b	L25
L27:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L25
L26:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	b	L25
L24:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	nop
L25:
	b	L122
L120:
	nop
	b	L119
L121:
	nop
	b	L119
L122:
	nop
	b	L119
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L30
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L123
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L32
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L33
	b	L123
L32:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L34
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L124
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L36
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L124
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L37
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L124
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L38
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L39
	b	L124
L38:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	b	L35
L39:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L35
L37:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	b	L35
L36:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L35
L34:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
L35:
	b	L124
L33:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L40
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L125
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L42
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L125
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L43
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L125
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L44
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L45
	b	L125
L44:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L41
L45:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	b	L41
L43:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L41
L42:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	b	L41
L40:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	nop
L41:
	b	L125
L30:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L46
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L126
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L48
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L126
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L49
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L126
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L50
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L51
	b	L126
L50:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	b	L47
L51:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	b	L47
L49:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	b	L47
L48:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L47
L46:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
L47:
	b	L126
L124:
	nop
	b	L123
L125:
	nop
	b	L123
L126:
	nop
	b	L123
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L52
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L127
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L54
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L55
	b	L127
L54:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L56
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L128
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L58
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L128
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L59
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L128
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L60
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L61
	b	L128
L60:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L57
L61:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	b	L57
L59:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L57
L58:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	b	L57
L56:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	nop
L57:
	b	L128
L55:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L62
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L129
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L64
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L129
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L65
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L129
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L66
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L67
	b	L129
L66:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	b	L63
L67:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L63
L65:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	b	L63
L64:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L63
L62:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
L63:
	b	L129
L52:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L68
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L130
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L70
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L130
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L71
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L130
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L72
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L73
	b	L130
L72:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L69
L73:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	b	L69
L71:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L69
L70:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	b	L69
L68:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	nop
L69:
	b	L130
L128:
	nop
	b	L127
L129:
	nop
	b	L127
L130:
	nop
	b	L127
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L74
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L131
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L76
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L77
	b	L131
L76:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L78
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L132
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L80
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L132
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L81
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L132
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L82
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L83
	b	L132
L82:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	b	L79
L83:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L79
L81:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	b	L79
L80:
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	b	L79
L78:
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	nop
L79:
	b	L132
L77:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L84
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L133
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L86
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L133
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L87
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L133
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L88
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L89
	b	L133
L88:
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	b	L85
L89:
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	b	L85
L87:
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	b	L85
L86:
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	b	L85
L84:
	adrp	x0, lC54@PAGE
	add	x0, x0, lC54@PAGEOFF;
	bl	_puts
	nop
L85:
	b	L133
L74:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L90
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L134
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L92
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L134
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L93
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L134
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L94
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L95
	b	L134
L94:
	adrp	x0, lC55@PAGE
	add	x0, x0, lC55@PAGEOFF;
	bl	_puts
	b	L91
L95:
	adrp	x0, lC56@PAGE
	add	x0, x0, lC56@PAGEOFF;
	bl	_puts
	b	L91
L93:
	adrp	x0, lC57@PAGE
	add	x0, x0, lC57@PAGEOFF;
	bl	_puts
	b	L91
L92:
	adrp	x0, lC58@PAGE
	add	x0, x0, lC58@PAGEOFF;
	bl	_puts
	b	L91
L90:
	adrp	x0, lC59@PAGE
	add	x0, x0, lC59@PAGEOFF;
	bl	_puts
	nop
L91:
	b	L134
L132:
	nop
	b	L131
L133:
	nop
	b	L131
L134:
	nop
	b	L131
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L96
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L135
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L98
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L99
	b	L135
L98:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L100
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L136
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L102
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L136
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L103
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L136
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L104
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L105
	b	L136
L104:
	adrp	x0, lC60@PAGE
	add	x0, x0, lC60@PAGEOFF;
	bl	_puts
	b	L101
L105:
	adrp	x0, lC61@PAGE
	add	x0, x0, lC61@PAGEOFF;
	bl	_puts
	b	L101
L103:
	adrp	x0, lC62@PAGE
	add	x0, x0, lC62@PAGEOFF;
	bl	_puts
	b	L101
L102:
	adrp	x0, lC63@PAGE
	add	x0, x0, lC63@PAGEOFF;
	bl	_puts
	b	L101
L100:
	adrp	x0, lC64@PAGE
	add	x0, x0, lC64@PAGEOFF;
	bl	_puts
	nop
L101:
	b	L136
L99:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L106
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L137
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L108
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L137
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L109
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L137
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L110
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L111
	b	L137
L110:
	adrp	x0, lC65@PAGE
	add	x0, x0, lC65@PAGEOFF;
	bl	_puts
	b	L107
L111:
	adrp	x0, lC66@PAGE
	add	x0, x0, lC66@PAGEOFF;
	bl	_puts
	b	L107
L109:
	adrp	x0, lC67@PAGE
	add	x0, x0, lC67@PAGEOFF;
	bl	_puts
	b	L107
L108:
	adrp	x0, lC68@PAGE
	add	x0, x0, lC68@PAGEOFF;
	bl	_puts
	b	L107
L106:
	adrp	x0, lC69@PAGE
	add	x0, x0, lC69@PAGEOFF;
	bl	_puts
	nop
L107:
	b	L137
L96:
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L112
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L138
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L114
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L138
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L115
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L138
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L116
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L117
	b	L138
L116:
	adrp	x0, lC70@PAGE
	add	x0, x0, lC70@PAGEOFF;
	bl	_puts
	b	L113
L117:
	adrp	x0, lC71@PAGE
	add	x0, x0, lC71@PAGEOFF;
	bl	_puts
	b	L113
L115:
	adrp	x0, lC72@PAGE
	add	x0, x0, lC72@PAGEOFF;
	bl	_puts
	b	L113
L114:
	adrp	x0, lC73@PAGE
	add	x0, x0, lC73@PAGEOFF;
	bl	_puts
	b	L113
L112:
	adrp	x0, lC74@PAGE
	add	x0, x0, lC74@PAGEOFF;
	bl	_puts
	nop
L113:
	b	L138
L136:
	nop
	b	L135
L137:
	nop
	b	L135
L138:
	nop
	b	L135
L119:
	nop
	b	L3
L123:
	nop
	b	L3
L127:
	nop
	b	L3
L131:
	nop
	b	L3
L135:
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
