	.file	"Surajwaghmare35_C-Lang_Notes_flatten.c"
	.text
	.globl	d2
	.bss
	.align 8
	.type	d2, @object
	.size	d2, 12
d2:
	.zero	12
	.globl	_TIG_IZ_1igG_envp
	.align 8
	.type	_TIG_IZ_1igG_envp, @object
	.size	_TIG_IZ_1igG_envp, 8
_TIG_IZ_1igG_envp:
	.zero	8
	.globl	_TIG_IZ_1igG_argv
	.align 8
	.type	_TIG_IZ_1igG_argv, @object
	.size	_TIG_IZ_1igG_argv, 8
_TIG_IZ_1igG_argv:
	.zero	8
	.globl	_TIG_IZ_1igG_argc
	.align 4
	.type	_TIG_IZ_1igG_argc, @object
	.size	_TIG_IZ_1igG_argc, 4
_TIG_IZ_1igG_argc:
	.zero	4
	.text
	.globl	sort
	.type	sort, @function
sort:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -40(%rbp)
	movq	$0, -8(%rbp)
.L21:
	cmpq	$12, -8(%rbp)
	ja	.L22
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
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L22-.L4
	.long	.L23-.L4
	.long	.L8-.L4
	.long	.L22-.L4
	.long	.L7-.L4
	.long	.L22-.L4
	.long	.L22-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L3:
	addl	$1, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L14
.L11:
	addl	$1, -20(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L14
.L5:
	cmpl	$4, -20(%rbp)
	jg	.L15
	movq	$2, -8(%rbp)
	jmp	.L14
.L15:
	movq	$4, -8(%rbp)
	jmp	.L14
.L8:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -12(%rbp)
	movl	-16(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	-16(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-40(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	movl	-16(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movl	-12(%rbp), %eax
	movl	%eax, (%rdx)
	movq	$12, -8(%rbp)
	jmp	.L14
.L6:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movl	-16(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,4), %rcx
	movq	-40(%rbp), %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jle	.L17
	movq	$5, -8(%rbp)
	jmp	.L14
.L17:
	movq	$12, -8(%rbp)
	jmp	.L14
.L12:
	movl	$1, -20(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L14
.L7:
	cmpl	$3, -16(%rbp)
	jg	.L19
	movq	$10, -8(%rbp)
	jmp	.L14
.L19:
	movq	$1, -8(%rbp)
	jmp	.L14
.L10:
	movl	$0, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L14
.L22:
	nop
.L14:
	jmp	.L21
.L23:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	sort, .-sort
	.globl	swap
	.type	swap, @function
swap:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$2, -8(%rbp)
.L30:
	cmpq	$2, -8(%rbp)
	je	.L25
	cmpq	$2, -8(%rbp)
	ja	.L31
	cmpq	$0, -8(%rbp)
	je	.L32
	cmpq	$1, -8(%rbp)
	jne	.L31
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -12(%rbp)
	movq	-32(%rbp), %rax
	movl	(%rax), %edx
	movq	-24(%rbp), %rax
	movl	%edx, (%rax)
	movq	-32(%rbp), %rax
	movl	-12(%rbp), %edx
	movl	%edx, (%rax)
	movq	$0, -8(%rbp)
	jmp	.L28
.L25:
	movq	$1, -8(%rbp)
	jmp	.L28
.L31:
	nop
.L28:
	jmp	.L30
.L32:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	swap, .-swap
	.globl	length
	.type	length, @function
length:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$0, -8(%rbp)
.L43:
	cmpq	$5, -8(%rbp)
	je	.L34
	cmpq	$5, -8(%rbp)
	ja	.L45
	cmpq	$4, -8(%rbp)
	je	.L36
	cmpq	$4, -8(%rbp)
	ja	.L45
	cmpq	$0, -8(%rbp)
	je	.L37
	cmpq	$2, -8(%rbp)
	je	.L38
	jmp	.L45
.L36:
	movl	-12(%rbp), %eax
	jmp	.L44
.L34:
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L40
.L37:
	movl	$0, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L40
.L38:
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L41
	movq	$5, -8(%rbp)
	jmp	.L40
.L41:
	movq	$4, -8(%rbp)
	jmp	.L40
.L45:
	nop
.L40:
	jmp	.L43
.L44:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	length, .-length
	.section	.rodata
.LC0:
	.string	"%d"
.LC1:
	.string	"\nEnter ur name: "
.LC2:
	.string	"%s"
.LC3:
	.string	"\nSurajGW"
	.align 8
.LC4:
	.string	"\n\n1. Addition.\n2. Odd-Even.\n3. Print N Natural No.\n4. Array-Sum.\n5. String.\n6. Swap\n7. Sort\n8. StrLen & StrRev.\n9. Exit."
.LC5:
	.string	"\nEnter your choice number: "
.LC6:
	.string	"\nEnter Two No. to Add: "
.LC7:
	.string	"%d %d"
.LC8:
	.string	"\nSum of %d & %d is: %d"
.LC9:
	.string	"\nInvalid Choice"
.LC10:
	.string	"Suraj"
.LC11:
	.string	"Length of sting is: %d\n"
.LC12:
	.string	"Enter 10 No: "
.LC13:
	.string	"max no. is: %d\n"
.LC14:
	.string	"Entet two No, to swap: "
.LC15:
	.string	"%d%d"
.LC16:
	.string	"Swap Values is: b=%d & c=%d\n"
	.align 8
.LC17:
	.string	"Enter 9 Number to fill 2nd 3*3 matrix: "
.LC18:
	.string	" %d"
	.align 8
.LC20:
	.string	"Sum of above is: %d and Avg is: %f\n"
	.align 8
.LC21:
	.string	"\nEnter 9 Number to fill 1st 3*3 matrix: "
.LC22:
	.string	"Enter today's date: "
.LC23:
	.string	"%d/%d/%d"
.LC24:
	.string	"Date: %d/%d/%d\n"
.LC25:
	.string	"\nValue of ~a is: %d\n"
	.align 8
.LC26:
	.string	"\nEnter a number to chech max: "
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$424, %rsp
	.cfi_offset 3, -24
	movl	%edi, -404(%rbp)
	movq	%rsi, -416(%rbp)
	movq	%rdx, -424(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movl	$0, d2(%rip)
	movl	$0, 4+d2(%rip)
	movl	$0, 8+d2(%rip)
	nop
.L47:
	movq	$0, _TIG_IZ_1igG_envp(%rip)
	nop
.L48:
	movq	$0, _TIG_IZ_1igG_argv(%rip)
	nop
.L49:
	movl	$0, _TIG_IZ_1igG_argc(%rip)
	nop
	nop
.L50:
.L51:
#APP
# 564 "Surajwaghmare35_C-Lang_Notes.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-1igG--0
# 0 "" 2
#NO_APP
	movl	-404(%rbp), %eax
	movl	%eax, _TIG_IZ_1igG_argc(%rip)
	movq	-416(%rbp), %rax
	movq	%rax, _TIG_IZ_1igG_argv(%rip)
	movq	-424(%rbp), %rax
	movq	%rax, _TIG_IZ_1igG_envp(%rip)
	nop
	movq	$70, -336(%rbp)
.L135:
	cmpq	$94, -336(%rbp)
	ja	.L137
	movq	-336(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L54(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L54(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L54:
	.long	.L99-.L54
	.long	.L98-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L97-.L54
	.long	.L96-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L95-.L54
	.long	.L94-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L93-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L92-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L91-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L90-.L54
	.long	.L89-.L54
	.long	.L137-.L54
	.long	.L88-.L54
	.long	.L87-.L54
	.long	.L137-.L54
	.long	.L86-.L54
	.long	.L85-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L84-.L54
	.long	.L83-.L54
	.long	.L82-.L54
	.long	.L81-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L80-.L54
	.long	.L79-.L54
	.long	.L78-.L54
	.long	.L137-.L54
	.long	.L77-.L54
	.long	.L76-.L54
	.long	.L75-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L74-.L54
	.long	.L73-.L54
	.long	.L137-.L54
	.long	.L72-.L54
	.long	.L71-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L70-.L54
	.long	.L69-.L54
	.long	.L68-.L54
	.long	.L137-.L54
	.long	.L67-.L54
	.long	.L66-.L54
	.long	.L137-.L54
	.long	.L65-.L54
	.long	.L64-.L54
	.long	.L63-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L62-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L61-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L60-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L59-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L58-.L54
	.long	.L57-.L54
	.long	.L56-.L54
	.long	.L55-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L137-.L54
	.long	.L53-.L54
	.text
.L60:
	movl	-384(%rbp), %eax
	movl	%eax, -356(%rbp)
	movq	$44, -336(%rbp)
	jmp	.L100
.L90:
	movl	$0, -372(%rbp)
	movq	$28, -336(%rbp)
	jmp	.L100
.L74:
	cmpl	$2, -372(%rbp)
	jg	.L101
	movq	$87, -336(%rbp)
	jmp	.L100
.L101:
	movq	$46, -336(%rbp)
	jmp	.L100
.L97:
	cmpl	$2, -372(%rbp)
	jg	.L103
	movq	$94, -336(%rbp)
	jmp	.L100
.L103:
	movq	$36, -336(%rbp)
	jmp	.L100
.L69:
	leaq	-304(%rbp), %rax
	movq	%rax, %rdi
	call	input
	leaq	-304(%rbp), %rax
	movq	%rax, %rdi
	call	display
	leaq	-304(%rbp), %rax
	movq	%rax, %rdi
	call	sort
	leaq	-304(%rbp), %rax
	movq	%rax, %rdi
	call	display
	movq	$36, -336(%rbp)
	jmp	.L100
.L56:
	call	Nnatural
	movq	$36, -336(%rbp)
	jmp	.L100
.L71:
	movl	-380(%rbp), %eax
	cmpl	$9, %eax
	ja	.L105
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L107(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L107(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L107:
	.long	.L105-.L107
	.long	.L115-.L107
	.long	.L114-.L107
	.long	.L113-.L107
	.long	.L112-.L107
	.long	.L111-.L107
	.long	.L110-.L107
	.long	.L109-.L107
	.long	.L108-.L107
	.long	.L106-.L107
	.text
.L106:
	movq	$90, -336(%rbp)
	jmp	.L116
.L108:
	movq	$65, -336(%rbp)
	jmp	.L116
.L109:
	movq	$62, -336(%rbp)
	jmp	.L116
.L110:
	movq	$0, -336(%rbp)
	jmp	.L116
.L111:
	movq	$25, -336(%rbp)
	jmp	.L116
.L112:
	movq	$73, -336(%rbp)
	jmp	.L116
.L113:
	movq	$89, -336(%rbp)
	jmp	.L116
.L114:
	movq	$68, -336(%rbp)
	jmp	.L116
.L115:
	movq	$38, -336(%rbp)
	jmp	.L116
.L105:
	movq	$53, -336(%rbp)
	nop
.L116:
	jmp	.L100
.L86:
	leaq	-96(%rbp), %rdx
	movl	-372(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-372(%rbp), %eax
	cltq
	movl	-96(%rbp,%rax,4), %eax
	addl	%eax, -364(%rbp)
	addl	$1, -372(%rbp)
	movq	$88, -336(%rbp)
	jmp	.L100
.L93:
	movl	$0, -368(%rbp)
	movq	$47, -336(%rbp)
	jmp	.L100
.L64:
	movl	-372(%rbp), %eax
	cltq
	movzbl	-44(%rbp,%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	putchar@PLT
	addl	$1, -372(%rbp)
	movq	$28, -336(%rbp)
	jmp	.L100
.L95:
	addl	$1, -372(%rbp)
	movq	$5, -336(%rbp)
	jmp	.L100
.L98:
	movl	$10, %edi
	call	putchar@PLT
	addl	$1, -372(%rbp)
	movq	$4, -336(%rbp)
	jmp	.L100
.L61:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-34(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-34(%rbp), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$36, -336(%rbp)
	jmp	.L100
.L63:
	movq	$32, -336(%rbp)
	jmp	.L100
.L92:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -372(%rbp)
	movq	$20, -336(%rbp)
	jmp	.L100
.L53:
	movl	$0, -368(%rbp)
	movq	$29, -336(%rbp)
	jmp	.L100
.L83:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-380(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$56, -336(%rbp)
	jmp	.L100
.L65:
	call	isEven
	movq	$36, -336(%rbp)
	jmp	.L100
.L89:
	addl	$1, -372(%rbp)
	movq	$52, -336(%rbp)
	jmp	.L100
.L94:
	cmpl	$9, -360(%rbp)
	jbe	.L117
	movq	$43, -336(%rbp)
	jmp	.L100
.L117:
	movq	$61, -336(%rbp)
	jmp	.L100
.L68:
	cmpl	$2, -368(%rbp)
	jg	.L119
	movq	$42, -336(%rbp)
	jmp	.L100
.L119:
	movq	$26, -336(%rbp)
	jmp	.L100
.L85:
	movl	$-6, -376(%rbp)
	movl	$1, -340(%rbp)
	movl	$1, -364(%rbp)
	movb	$83, -44(%rbp)
	movb	$85, -43(%rbp)
	movb	$82, -42(%rbp)
	movb	$65, -41(%rbp)
	movb	$74, -40(%rbp)
	movb	$0, -39(%rbp)
	movl	$6, -360(%rbp)
	movq	$9, -336(%rbp)
	jmp	.L100
.L55:
	movl	$0, %edi
	call	exit@PLT
.L72:
	movl	-388(%rbp), %edx
	movl	-384(%rbp), %eax
	cmpl	%eax, %edx
	jle	.L121
	movq	$48, -336(%rbp)
	jmp	.L100
.L121:
	movq	$80, -336(%rbp)
	jmp	.L100
.L81:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-384(%rbp), %rdx
	leaq	-388(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-384(%rbp), %edx
	movl	-388(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	add
	movl	%eax, -344(%rbp)
	movl	-384(%rbp), %edx
	movl	-388(%rbp), %eax
	movl	-344(%rbp), %ecx
	movl	%eax, %esi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$36, -336(%rbp)
	jmp	.L100
.L70:
	movl	-360(%rbp), %eax
	movb	$0, -44(%rbp,%rax)
	addl	$1, -360(%rbp)
	movq	$9, -336(%rbp)
	jmp	.L100
.L58:
	movl	$0, -368(%rbp)
	movq	$63, -336(%rbp)
	jmp	.L100
.L59:
	leaq	-192(%rbp), %rcx
	movl	-368(%rbp), %eax
	movslq	%eax, %rsi
	movl	-372(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	addq	%rsi, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -368(%rbp)
	movq	$47, -336(%rbp)
	jmp	.L100
.L75:
	movl	-388(%rbp), %eax
	movl	%eax, -356(%rbp)
	movq	$44, -336(%rbp)
	jmp	.L100
.L88:
	movl	-372(%rbp), %eax
	cltq
	movzbl	-44(%rbp,%rax), %eax
	testb	%al, %al
	je	.L123
	movq	$69, -336(%rbp)
	jmp	.L100
.L123:
	movq	$77, -336(%rbp)
	jmp	.L100
.L73:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$36, -336(%rbp)
	jmp	.L100
.L67:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	length
	movl	%eax, -348(%rbp)
	movl	-348(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$36, -336(%rbp)
	jmp	.L100
.L76:
	cmpl	$2, -368(%rbp)
	jg	.L125
	movq	$84, -336(%rbp)
	jmp	.L100
.L125:
	movq	$8, -336(%rbp)
	jmp	.L100
.L62:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -372(%rbp)
	movq	$88, -336(%rbp)
	jmp	.L100
.L78:
	movl	-356(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$1, -372(%rbp)
	movq	$20, -336(%rbp)
	jmp	.L100
.L96:
	cmpl	$2, -372(%rbp)
	jg	.L127
	movq	$12, -336(%rbp)
	jmp	.L100
.L127:
	movq	$37, -336(%rbp)
	jmp	.L100
.L82:
	movl	$0, -372(%rbp)
	movq	$4, -336(%rbp)
	jmp	.L100
.L80:
	leaq	-240(%rbp), %rcx
	movl	-368(%rbp), %eax
	movslq	%eax, %rsi
	movl	-372(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	addq	%rsi, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -368(%rbp)
	movq	$63, -336(%rbp)
	jmp	.L100
.L99:
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-384(%rbp), %rdx
	leaq	-388(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-384(%rbp), %rdx
	leaq	-388(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	swap
	movl	-384(%rbp), %edx
	movl	-388(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$36, -336(%rbp)
	jmp	.L100
.L77:
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -372(%rbp)
	movq	$5, -336(%rbp)
	jmp	.L100
.L66:
	movl	-368(%rbp), %eax
	movslq	%eax, %rcx
	movl	-372(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	addq	%rcx, %rax
	movl	-240(%rbp,%rax,4), %ecx
	movl	-368(%rbp), %eax
	movslq	%eax, %rsi
	movl	-372(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	addq	%rsi, %rax
	movl	-192(%rbp,%rax,4), %eax
	addl	%ecx, %eax
	movl	%eax, -352(%rbp)
	movl	-368(%rbp), %eax
	movslq	%eax, %rcx
	movl	-372(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	leaq	(%rax,%rcx), %rdx
	movl	-352(%rbp), %eax
	movl	%eax, -144(%rbp,%rdx,4)
	movl	-352(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -368(%rbp)
	movq	$29, -336(%rbp)
	jmp	.L100
.L57:
	cmpl	$9, -372(%rbp)
	jg	.L129
	movq	$31, -336(%rbp)
	jmp	.L100
.L129:
	movq	$35, -336(%rbp)
	jmp	.L100
.L84:
	pxor	%xmm0, %xmm0
	cvtsi2sdl	-364(%rbp), %xmm0
	movsd	.LC19(%rip), %xmm1
	divsd	%xmm1, %xmm0
	movq	%xmm0, %rdx
	movl	-364(%rbp), %eax
	movq	%rdx, %xmm0
	movl	%eax, %esi
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -372(%rbp)
	movq	$52, -336(%rbp)
	jmp	.L100
.L87:
	cmpl	$2, -368(%rbp)
	jg	.L131
	movq	$66, -336(%rbp)
	jmp	.L100
.L131:
	movq	$1, -336(%rbp)
	jmp	.L100
.L79:
	movl	$15, -328(%rbp)
	movl	$9, -324(%rbp)
	movl	$2024, -320(%rbp)
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-316(%rbp), %rax
	leaq	8(%rax), %rcx
	leaq	-316(%rbp), %rax
	leaq	4(%rax), %rdx
	leaq	-316(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC23(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-308(%rbp), %ecx
	movl	-312(%rbp), %edx
	movl	-316(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC24(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-272(%rbp), %rax
	movq	%rax, %rdi
	call	struct_input
	subq	$32, %rsp
	movq	%rsp, %rax
	movq	-272(%rbp), %rcx
	movq	-264(%rbp), %rbx
	movq	%rcx, (%rax)
	movq	%rbx, 8(%rax)
	movq	-256(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movl	-248(%rbp), %edx
	movl	%edx, 24(%rax)
	call	struct_display
	addq	$32, %rsp
	movl	-376(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC25(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC26(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-384(%rbp), %rdx
	leaq	-388(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$55, -336(%rbp)
	jmp	.L100
.L91:
	cmpl	$5, -372(%rbp)
	jg	.L133
	movq	$16, -336(%rbp)
	jmp	.L100
.L133:
	movq	$36, -336(%rbp)
	jmp	.L100
.L137:
	nop
.L100:
	jmp	.L135
	.cfi_endproc
.LFE3:
	.size	main, .-main
	.section	.rodata
.LC27:
	.string	"\nEnter five No. to Sort: "
	.text
	.globl	input
	.type	input, @function
input:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$6, -8(%rbp)
.L150:
	cmpq	$7, -8(%rbp)
	ja	.L151
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L141(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L141(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L141:
	.long	.L145-.L141
	.long	.L144-.L141
	.long	.L143-.L141
	.long	.L151-.L141
	.long	.L151-.L141
	.long	.L151-.L141
	.long	.L142-.L141
	.long	.L152-.L141
	.text
.L144:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L146
.L142:
	movq	$0, -8(%rbp)
	jmp	.L146
.L145:
	leaq	.LC27(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L146
.L143:
	cmpl	$4, -12(%rbp)
	jg	.L148
	movq	$1, -8(%rbp)
	jmp	.L146
.L148:
	movq	$7, -8(%rbp)
	jmp	.L146
.L151:
	nop
.L146:
	jmp	.L150
.L152:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	input, .-input
	.section	.rodata
	.align 8
.LC28:
	.string	"\nEnter BookID, Title & Price: "
.LC29:
	.string	"%d %s %f"
	.text
	.globl	struct_input
	.type	struct_input, @function
struct_input:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$88, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -88(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	$2, -72(%rbp)
.L159:
	cmpq	$2, -72(%rbp)
	je	.L154
	cmpq	$2, -72(%rbp)
	ja	.L162
	cmpq	$0, -72(%rbp)
	je	.L156
	cmpq	$1, -72(%rbp)
	jne	.L162
	leaq	.LC28(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-64(%rbp), %rax
	leaq	24(%rax), %rcx
	leaq	-64(%rbp), %rax
	leaq	4(%rax), %rdx
	leaq	-64(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC29(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	$0, -72(%rbp)
	jmp	.L157
.L156:
	movq	-88(%rbp), %rax
	movq	-64(%rbp), %rcx
	movq	-56(%rbp), %rbx
	movq	%rcx, (%rax)
	movq	%rbx, 8(%rax)
	movq	-48(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movl	-40(%rbp), %edx
	movl	%edx, 24(%rax)
	movq	-24(%rbp), %rax
	subq	%fs:40, %rax
	je	.L160
	jmp	.L161
.L154:
	movq	$1, -72(%rbp)
	jmp	.L157
.L162:
	nop
.L157:
	jmp	.L159
.L161:
	call	__stack_chk_fail@PLT
.L160:
	movq	-88(%rbp), %rax
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	struct_input, .-struct_input
	.globl	add
	.type	add, @function
add:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$0, -8(%rbp)
.L166:
	cmpq	$0, -8(%rbp)
	jne	.L169
	movl	-20(%rbp), %edx
	movl	-24(%rbp), %eax
	addl	%edx, %eax
	jmp	.L168
.L169:
	nop
	jmp	.L166
.L168:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	add, .-add
	.section	.rodata
.LC30:
	.string	"%d "
	.align 8
.LC31:
	.string	"\nEnter a no. to print N Narural: "
.LC32:
	.string	"\nN narural no. of %d is: "
	.text
	.globl	Nnatural
	.type	Nnatural, @function
Nnatural:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, -16(%rbp)
.L182:
	cmpq	$7, -16(%rbp)
	ja	.L185
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L173(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L173(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L173:
	.long	.L177-.L173
	.long	.L186-.L173
	.long	.L185-.L173
	.long	.L175-.L173
	.long	.L185-.L173
	.long	.L185-.L173
	.long	.L174-.L173
	.long	.L172-.L173
	.text
.L175:
	movl	-24(%rbp), %eax
	cmpl	%eax, -20(%rbp)
	jg	.L179
	movq	$6, -16(%rbp)
	jmp	.L181
.L179:
	movq	$1, -16(%rbp)
	jmp	.L181
.L174:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC30(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -20(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L181
.L177:
	movq	$7, -16(%rbp)
	jmp	.L181
.L172:
	leaq	.LC31(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC32(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$1, -20(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L181
.L185:
	nop
.L181:
	jmp	.L182
.L186:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L184
	call	__stack_chk_fail@PLT
.L184:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	Nnatural, .-Nnatural
	.section	.rodata
	.align 8
.LC33:
	.string	"BookID= %d, Title= %s & Price= %f\n"
	.text
	.globl	struct_display
	.type	struct_display, @function
struct_display:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L192:
	cmpq	$0, -8(%rbp)
	je	.L193
	cmpq	$1, -8(%rbp)
	jne	.L194
	movss	40(%rbp), %xmm0
	pxor	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	movq	%xmm1, %rcx
	movl	16(%rbp), %eax
	leaq	20(%rbp), %rdx
	movq	%rcx, %xmm0
	movl	%eax, %esi
	leaq	.LC33(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$0, -8(%rbp)
	jmp	.L190
.L194:
	nop
.L190:
	jmp	.L192
.L193:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	struct_display, .-struct_display
	.section	.rodata
.LC34:
	.string	"\nNo. %d is even."
.LC35:
	.string	"\nNo. %d is odd."
	.align 8
.LC36:
	.string	"\nEnter a No. to check odd or even: "
	.text
	.globl	isEven
	.type	isEven, @function
isEven:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$2, -16(%rbp)
.L208:
	cmpq	$5, -16(%rbp)
	ja	.L211
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L198(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L198(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L198:
	.long	.L203-.L198
	.long	.L202-.L198
	.long	.L201-.L198
	.long	.L212-.L198
	.long	.L199-.L198
	.long	.L197-.L198
	.text
.L199:
	movl	-20(%rbp), %eax
	andl	$1, %eax
	testl	%eax, %eax
	jne	.L204
	movq	$1, -16(%rbp)
	jmp	.L206
.L204:
	movq	$5, -16(%rbp)
	jmp	.L206
.L202:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC34(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L206
.L197:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC35(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L206
.L203:
	leaq	.LC36(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$4, -16(%rbp)
	jmp	.L206
.L201:
	movq	$0, -16(%rbp)
	jmp	.L206
.L211:
	nop
.L206:
	jmp	.L208
.L212:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L210
	call	__stack_chk_fail@PLT
.L210:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	isEven, .-isEven
	.globl	display
	.type	display, @function
display:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$2, -8(%rbp)
.L222:
	cmpq	$6, -8(%rbp)
	je	.L223
	cmpq	$6, -8(%rbp)
	ja	.L224
	cmpq	$5, -8(%rbp)
	je	.L216
	cmpq	$5, -8(%rbp)
	ja	.L224
	cmpq	$2, -8(%rbp)
	je	.L217
	cmpq	$3, -8(%rbp)
	jne	.L224
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC30(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L218
.L216:
	cmpl	$4, -12(%rbp)
	jg	.L220
	movq	$3, -8(%rbp)
	jmp	.L218
.L220:
	movq	$6, -8(%rbp)
	jmp	.L218
.L217:
	movl	$0, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L218
.L224:
	nop
.L218:
	jmp	.L222
.L223:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	display, .-display
	.globl	reverse
	.type	reverse, @function
reverse:
.LFB16:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -40(%rbp)
	movq	$2, -8(%rbp)
.L241:
	cmpq	$10, -8(%rbp)
	ja	.L243
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L228(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L228(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L228:
	.long	.L243-.L228
	.long	.L243-.L228
	.long	.L234-.L228
	.long	.L233-.L228
	.long	.L232-.L228
	.long	.L231-.L228
	.long	.L243-.L228
	.long	.L243-.L228
	.long	.L230-.L228
	.long	.L229-.L228
	.long	.L227-.L228
	.text
.L232:
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L235
	movq	$5, -8(%rbp)
	jmp	.L237
.L235:
	movq	$8, -8(%rbp)
	jmp	.L237
.L230:
	movl	$0, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L237
.L233:
	movl	-16(%rbp), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	cmpl	%eax, -12(%rbp)
	jge	.L238
	movq	$10, -8(%rbp)
	jmp	.L237
.L238:
	movq	$9, -8(%rbp)
	jmp	.L237
.L229:
	movq	-40(%rbp), %rax
	jmp	.L242
.L231:
	addl	$1, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L237
.L227:
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movb	%al, -17(%rbp)
	movl	-16(%rbp), %eax
	cltq
	movl	-12(%rbp), %edx
	movslq	%edx, %rdx
	subq	%rdx, %rax
	leaq	-1(%rax), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	-12(%rbp), %edx
	movslq	%edx, %rcx
	movq	-40(%rbp), %rdx
	addq	%rcx, %rdx
	movzbl	(%rax), %eax
	movb	%al, (%rdx)
	movl	-16(%rbp), %eax
	cltq
	movl	-12(%rbp), %edx
	movslq	%edx, %rdx
	subq	%rdx, %rax
	leaq	-1(%rax), %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movzbl	-17(%rbp), %eax
	movb	%al, (%rdx)
	addl	$1, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L237
.L234:
	movl	$0, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L237
.L243:
	nop
.L237:
	jmp	.L241
.L242:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE16:
	.size	reverse, .-reverse
	.section	.rodata
	.align 8
.LC19:
	.long	0
	.long	1076101120
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
