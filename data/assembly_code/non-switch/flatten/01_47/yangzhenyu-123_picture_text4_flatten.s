	.file	"yangzhenyu-123_picture_text4_flatten.c"
	.text
	.globl	_TIG_IZ_3W8p_argv
	.bss
	.align 8
	.type	_TIG_IZ_3W8p_argv, @object
	.size	_TIG_IZ_3W8p_argv, 8
_TIG_IZ_3W8p_argv:
	.zero	8
	.globl	g_pEnd
	.align 8
	.type	g_pEnd, @object
	.size	g_pEnd, 8
g_pEnd:
	.zero	8
	.globl	g_pHead
	.align 8
	.type	g_pHead, @object
	.size	g_pHead, 8
g_pHead:
	.zero	8
	.globl	_TIG_IZ_3W8p_envp
	.align 8
	.type	_TIG_IZ_3W8p_envp, @object
	.size	_TIG_IZ_3W8p_envp, 8
_TIG_IZ_3W8p_envp:
	.zero	8
	.globl	_TIG_IZ_3W8p_argc
	.align 4
	.type	_TIG_IZ_3W8p_argc, @object
	.size	_TIG_IZ_3W8p_argc, 4
_TIG_IZ_3W8p_argc:
	.zero	4
	.text
	.globl	SleectList
	.type	SleectList, @function
SleectList:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$4, -8(%rbp)
.L16:
	cmpq	$7, -8(%rbp)
	ja	.L17
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L17-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L17-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L6:
	movq	g_pHead(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L10
.L8:
	movq	-16(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	cmpq	%rax, %rdx
	jne	.L11
	movq	$7, -8(%rbp)
	jmp	.L10
.L11:
	movq	$3, -8(%rbp)
	jmp	.L10
.L7:
	movq	-16(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L10
.L5:
	movl	$0, %eax
	jmp	.L13
.L9:
	cmpq	$0, -16(%rbp)
	je	.L14
	movq	$1, -8(%rbp)
	jmp	.L10
.L14:
	movq	$6, -8(%rbp)
	jmp	.L10
.L3:
	movq	-16(%rbp), %rax
	jmp	.L13
.L17:
	nop
.L10:
	jmp	.L16
.L13:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	SleectList, .-SleectList
	.globl	FreeList
	.type	FreeList, @function
FreeList:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$5, -16(%rbp)
.L30:
	cmpq	$7, -16(%rbp)
	ja	.L31
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L21(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L21(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L21:
	.long	.L25-.L21
	.long	.L31-.L21
	.long	.L32-.L21
	.long	.L23-.L21
	.long	.L31-.L21
	.long	.L22-.L21
	.long	.L31-.L21
	.long	.L20-.L21
	.text
.L23:
	movq	-24(%rbp), %rax
	movq	%rax, -8(%rbp)
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$7, -16(%rbp)
	jmp	.L26
.L22:
	movq	g_pHead(%rip), %rax
	movq	%rax, -24(%rbp)
	movq	$7, -16(%rbp)
	jmp	.L26
.L25:
	movq	$0, g_pHead(%rip)
	movq	$0, g_pEnd(%rip)
	movq	$2, -16(%rbp)
	jmp	.L26
.L20:
	cmpq	$0, -24(%rbp)
	je	.L27
	movq	$3, -16(%rbp)
	jmp	.L26
.L27:
	movq	$0, -16(%rbp)
	jmp	.L26
.L31:
	nop
.L26:
	jmp	.L30
.L32:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	FreeList, .-FreeList
	.globl	AddListHead
	.type	AddListHead, @function
AddListHead:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	$2, -16(%rbp)
.L46:
	cmpq	$7, -16(%rbp)
	ja	.L47
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L36(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L36(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L36:
	.long	.L41-.L36
	.long	.L40-.L36
	.long	.L39-.L36
	.long	.L38-.L36
	.long	.L47-.L36
	.long	.L47-.L36
	.long	.L48-.L36
	.long	.L35-.L36
	.text
.L40:
	movq	-24(%rbp), %rax
	movq	%rax, g_pHead(%rip)
	movq	-24(%rbp), %rax
	movq	%rax, g_pEnd(%rip)
	movq	$6, -16(%rbp)
	jmp	.L42
.L38:
	movq	g_pHead(%rip), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, 16(%rax)
	movq	-24(%rbp), %rax
	movq	%rax, g_pHead(%rip)
	movq	$6, -16(%rbp)
	jmp	.L42
.L41:
	movl	$24, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-24(%rbp), %rax
	movq	-48(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-24(%rbp), %rax
	movq	$0, 16(%rax)
	movq	$7, -16(%rbp)
	jmp	.L42
.L35:
	movq	g_pHead(%rip), %rax
	testq	%rax, %rax
	jne	.L44
	movq	$1, -16(%rbp)
	jmp	.L42
.L44:
	movq	$3, -16(%rbp)
	jmp	.L42
.L39:
	movq	$0, -16(%rbp)
	jmp	.L42
.L47:
	nop
.L42:
	jmp	.L46
.L48:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	AddListHead, .-AddListHead
	.section	.rodata
.LC0:
	.string	"\346\262\241\346\234\211\350\277\231\344\270\252\346\225\260\346\215\256\357\274\201"
.LC1:
	.string	"\345\267\262\347\273\217\346\211\276\345\210\260\346\225\260\346\215\256\357\274\232%s\n"
.LC2:
	.string	"input name:"
.LC3:
	.string	"%s"
.LC4:
	.string	"input number: "
.LC5:
	.string	"input location:"
.LC6:
	.string	"input Number:"
.LC7:
	.string	"input Name:"
.LC8:
	.string	"No.2"
	.text
	.globl	main
	.type	main, @function
main:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$336, %rsp
	movl	%edi, -308(%rbp)
	movq	%rsi, -320(%rbp)
	movq	%rdx, -328(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, g_pEnd(%rip)
	nop
.L50:
	movq	$0, g_pHead(%rip)
	nop
.L51:
	movq	$0, _TIG_IZ_3W8p_envp(%rip)
	nop
.L52:
	movq	$0, _TIG_IZ_3W8p_argv(%rip)
	nop
.L53:
	movl	$0, _TIG_IZ_3W8p_argc(%rip)
	nop
	nop
.L54:
.L55:
#APP
# 190 "yangzhenyu-123_picture_text4.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-3W8p--0
# 0 "" 2
#NO_APP
	movl	-308(%rbp), %eax
	movl	%eax, _TIG_IZ_3W8p_argc(%rip)
	movq	-320(%rbp), %rax
	movq	%rax, _TIG_IZ_3W8p_argv(%rip)
	movq	-328(%rbp), %rax
	movq	%rax, _TIG_IZ_3W8p_envp(%rip)
	nop
	movq	$11, -288(%rbp)
.L78:
	cmpq	$18, -288(%rbp)
	ja	.L81
	movq	-288(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L58(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L58(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L58:
	.long	.L69-.L58
	.long	.L68-.L58
	.long	.L81-.L58
	.long	.L81-.L58
	.long	.L67-.L58
	.long	.L81-.L58
	.long	.L66-.L58
	.long	.L81-.L58
	.long	.L65-.L58
	.long	.L64-.L58
	.long	.L81-.L58
	.long	.L63-.L58
	.long	.L81-.L58
	.long	.L81-.L58
	.long	.L62-.L58
	.long	.L61-.L58
	.long	.L60-.L58
	.long	.L59-.L58
	.long	.L57-.L58
	.text
.L57:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$8, -288(%rbp)
	jmp	.L70
.L67:
	movq	-296(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -288(%rbp)
	jmp	.L70
.L62:
	movl	$0, -300(%rbp)
	movq	$6, -288(%rbp)
	jmp	.L70
.L61:
	movl	$1, %edi
	call	malloc@PLT
	movq	%rax, -280(%rbp)
	movl	-304(%rbp), %eax
	cltq
	movq	-280(%rbp), %rdx
	movq	%rdx, -208(%rbp,%rax,8)
	movl	$1, %edi
	call	malloc@PLT
	movq	%rax, -272(%rbp)
	movl	-304(%rbp), %eax
	cltq
	movq	-272(%rbp), %rdx
	movq	%rdx, -112(%rbp,%rax,8)
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	-304(%rbp), %eax
	cltq
	movq	-208(%rbp,%rax,8), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	-304(%rbp), %eax
	cltq
	movq	-112(%rbp,%rax,8), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -304(%rbp)
	movq	$16, -288(%rbp)
	jmp	.L70
.L65:
	movl	$1, %edi
	call	malloc@PLT
	movq	%rax, -264(%rbp)
	movq	-264(%rbp), %rax
	movq	%rax, -256(%rbp)
	movl	$1, %edi
	call	malloc@PLT
	movq	%rax, -248(%rbp)
	movq	-248(%rbp), %rax
	movq	%rax, -240(%rbp)
	movl	$1, %edi
	call	malloc@PLT
	movq	%rax, -232(%rbp)
	movq	-232(%rbp), %rax
	movq	%rax, -224(%rbp)
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-256(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-240(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-224(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	-224(%rbp), %rdx
	movq	-240(%rbp), %rcx
	movq	-256(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	AddListAppoint
	call	ScanList
	call	FreeList
	movq	$9, -288(%rbp)
	jmp	.L70
.L68:
	movl	-300(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	-208(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	movl	-300(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rcx
	leaq	-112(%rbp), %rax
	addq	%rcx, %rax
	movq	(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	AddListHead
	addl	$1, -300(%rbp)
	movq	$6, -288(%rbp)
	jmp	.L70
.L60:
	cmpl	$2, -304(%rbp)
	jg	.L71
	movq	$15, -288(%rbp)
	jmp	.L70
.L71:
	movq	$14, -288(%rbp)
	jmp	.L70
.L63:
	movl	$0, -304(%rbp)
	movq	$16, -288(%rbp)
	jmp	.L70
.L64:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L79
	jmp	.L80
.L59:
	cmpq	$0, -296(%rbp)
	je	.L74
	movq	$4, -288(%rbp)
	jmp	.L70
.L74:
	movq	$18, -288(%rbp)
	jmp	.L70
.L66:
	cmpl	$2, -300(%rbp)
	jg	.L76
	movq	$1, -288(%rbp)
	jmp	.L70
.L76:
	movq	$0, -288(%rbp)
	jmp	.L70
.L69:
	call	ScanList
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	SleectList
	movq	%rax, -216(%rbp)
	movq	-216(%rbp), %rax
	movq	%rax, -296(%rbp)
	movq	$17, -288(%rbp)
	jmp	.L70
.L81:
	nop
.L70:
	jmp	.L78
.L80:
	call	__stack_chk_fail@PLT
.L79:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.section	.rodata
.LC9:
	.string	"%s\t"
	.text
	.globl	ScanList
	.type	ScanList, @function
ScanList:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$5, -8(%rbp)
.L91:
	cmpq	$6, -8(%rbp)
	je	.L83
	cmpq	$6, -8(%rbp)
	ja	.L92
	cmpq	$5, -8(%rbp)
	je	.L85
	cmpq	$5, -8(%rbp)
	ja	.L92
	cmpq	$1, -8(%rbp)
	je	.L93
	cmpq	$4, -8(%rbp)
	jne	.L92
	cmpq	$0, -16(%rbp)
	je	.L87
	movq	$6, -8(%rbp)
	jmp	.L89
.L87:
	movq	$1, -8(%rbp)
	jmp	.L89
.L83:
	movq	-16(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-16(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-16(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -16(%rbp)
	movl	$10, %edi
	call	putchar@PLT
	movq	$4, -8(%rbp)
	jmp	.L89
.L85:
	movq	g_pHead(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L89
.L92:
	nop
.L89:
	jmp	.L91
.L93:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	ScanList, .-ScanList
	.globl	AddListEnd
	.type	AddListEnd, @function
AddListEnd:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	$5, -16(%rbp)
.L107:
	cmpq	$7, -16(%rbp)
	ja	.L108
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L97(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L97(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L97:
	.long	.L108-.L97
	.long	.L108-.L97
	.long	.L109-.L97
	.long	.L101-.L97
	.long	.L100-.L97
	.long	.L99-.L97
	.long	.L98-.L97
	.long	.L96-.L97
	.text
.L100:
	movq	-24(%rbp), %rax
	movq	%rax, g_pHead(%rip)
	movq	-24(%rbp), %rax
	movq	%rax, g_pEnd(%rip)
	movq	$2, -16(%rbp)
	jmp	.L103
.L101:
	movq	g_pEnd(%rip), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	-24(%rbp), %rax
	movq	%rax, g_pEnd(%rip)
	movq	$2, -16(%rbp)
	jmp	.L103
.L98:
	movl	$24, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-24(%rbp), %rax
	movq	-48(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-24(%rbp), %rax
	movq	$0, 16(%rax)
	movq	$7, -16(%rbp)
	jmp	.L103
.L99:
	movq	$6, -16(%rbp)
	jmp	.L103
.L96:
	movq	g_pHead(%rip), %rax
	testq	%rax, %rax
	jne	.L104
	movq	$4, -16(%rbp)
	jmp	.L103
.L104:
	movq	$3, -16(%rbp)
	jmp	.L103
.L108:
	nop
.L103:
	jmp	.L107
.L109:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	AddListEnd, .-AddListEnd
	.section	.rodata
.LC10:
	.string	"\350\257\245\351\223\276\350\241\250\344\270\272\347\251\272\357\274\201"
.LC11:
	.string	"\346\262\241\346\234\211\346\214\207\345\256\232\347\273\223\347\202\271\357\274\201"
	.text
	.globl	AddListAppoint
	.type	AddListAppoint, @function
AddListAppoint:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -56(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	$3, -24(%rbp)
.L133:
	cmpq	$15, -24(%rbp)
	ja	.L134
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L113(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L113(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L113:
	.long	.L124-.L113
	.long	.L134-.L113
	.long	.L135-.L113
	.long	.L122-.L113
	.long	.L134-.L113
	.long	.L121-.L113
	.long	.L120-.L113
	.long	.L119-.L113
	.long	.L134-.L113
	.long	.L118-.L113
	.long	.L117-.L113
	.long	.L116-.L113
	.long	.L134-.L113
	.long	.L135-.L113
	.long	.L135-.L113
	.long	.L112-.L113
	.text
.L112:
	cmpq	$0, -40(%rbp)
	jne	.L126
	movq	$7, -24(%rbp)
	jmp	.L128
.L126:
	movq	$6, -24(%rbp)
	jmp	.L128
.L122:
	movq	g_pHead(%rip), %rax
	testq	%rax, %rax
	jne	.L129
	movq	$0, -24(%rbp)
	jmp	.L128
.L129:
	movq	$5, -24(%rbp)
	jmp	.L128
.L116:
	movq	-40(%rbp), %rax
	movq	16(%rax), %rdx
	movq	-32(%rbp), %rax
	movq	%rdx, 16(%rax)
	movq	-40(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	$2, -24(%rbp)
	jmp	.L128
.L118:
	movq	-40(%rbp), %rax
	movq	g_pEnd(%rip), %rdx
	cmpq	%rdx, %rax
	jne	.L131
	movq	$10, -24(%rbp)
	jmp	.L128
.L131:
	movq	$11, -24(%rbp)
	jmp	.L128
.L120:
	movl	$24, %edi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	-64(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-32(%rbp), %rax
	movq	-72(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-32(%rbp), %rax
	movq	$0, 16(%rax)
	movq	$9, -24(%rbp)
	jmp	.L128
.L121:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	SleectList
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	$15, -24(%rbp)
	jmp	.L128
.L117:
	movq	g_pEnd(%rip), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	-32(%rbp), %rax
	movq	%rax, g_pEnd(%rip)
	movq	$2, -24(%rbp)
	jmp	.L128
.L124:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$13, -24(%rbp)
	jmp	.L128
.L119:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$14, -24(%rbp)
	jmp	.L128
.L134:
	nop
.L128:
	jmp	.L133
.L135:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	AddListAppoint, .-AddListAppoint
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
