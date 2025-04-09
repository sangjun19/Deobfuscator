	.file	"ar-zoop_DSA_queue_flatten.c"
	.text
	.globl	_TIG_IZ_pwXr_argv
	.bss
	.align 8
	.type	_TIG_IZ_pwXr_argv, @object
	.size	_TIG_IZ_pwXr_argv, 8
_TIG_IZ_pwXr_argv:
	.zero	8
	.globl	_TIG_IZ_pwXr_envp
	.align 8
	.type	_TIG_IZ_pwXr_envp, @object
	.size	_TIG_IZ_pwXr_envp, 8
_TIG_IZ_pwXr_envp:
	.zero	8
	.globl	r
	.align 4
	.type	r, @object
	.size	r, 4
r:
	.zero	4
	.globl	_TIG_IZ_pwXr_argc
	.align 4
	.type	_TIG_IZ_pwXr_argc, @object
	.size	_TIG_IZ_pwXr_argc, 4
_TIG_IZ_pwXr_argc:
	.zero	4
	.globl	f
	.align 4
	.type	f, @object
	.size	f, 4
f:
	.zero	4
	.globl	q
	.align 32
	.type	q, @object
	.size	q, 40
q:
	.zero	40
	.section	.rodata
.LC0:
	.string	"Done"
.LC1:
	.string	"Underflow"
	.text
	.globl	dequeue
	.type	dequeue, @function
dequeue:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$2, -8(%rbp)
.L11:
	cmpq	$4, -8(%rbp)
	je	.L2
	cmpq	$4, -8(%rbp)
	ja	.L12
	cmpq	$3, -8(%rbp)
	je	.L13
	cmpq	$3, -8(%rbp)
	ja	.L12
	cmpq	$0, -8(%rbp)
	je	.L5
	cmpq	$2, -8(%rbp)
	je	.L6
	jmp	.L12
.L2:
	movl	f(%rip), %eax
	addl	$1, %eax
	movl	%eax, f(%rip)
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -8(%rbp)
	jmp	.L7
.L5:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -8(%rbp)
	jmp	.L7
.L6:
	movl	f(%rip), %eax
	cmpl	$-1, %eax
	jne	.L9
	movq	$0, -8(%rbp)
	jmp	.L7
.L9:
	movq	$4, -8(%rbp)
	jmp	.L7
.L12:
	nop
.L7:
	jmp	.L11
.L13:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	dequeue, .-dequeue
	.section	.rodata
.LC2:
	.string	"\nBye"
.LC3:
	.string	"choose an option: "
	.align 8
.LC4:
	.string	"1) enqueue\n2) dequeue\n3)display\n4)exit: "
.LC5:
	.string	"%d"
.LC6:
	.string	"Enter ele: "
.LC7:
	.string	"Enter correct option!"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, -20(%rbp)
	jmp	.L15
.L16:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	q(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -20(%rbp)
.L15:
	cmpl	$9, -20(%rbp)
	jle	.L16
	nop
.L17:
	movl	$-1, r(%rip)
	nop
.L18:
	movl	$-1, f(%rip)
	nop
.L19:
	movq	$0, _TIG_IZ_pwXr_envp(%rip)
	nop
.L20:
	movq	$0, _TIG_IZ_pwXr_argv(%rip)
	nop
.L21:
	movl	$0, _TIG_IZ_pwXr_argc(%rip)
	nop
	nop
.L22:
.L23:
#APP
# 159 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-pwXr--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_pwXr_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_pwXr_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_pwXr_envp(%rip)
	nop
	movq	$15, -16(%rbp)
.L41:
	cmpq	$15, -16(%rbp)
	ja	.L43
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L26(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L26(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L26:
	.long	.L33-.L26
	.long	.L32-.L26
	.long	.L31-.L26
	.long	.L30-.L26
	.long	.L29-.L26
	.long	.L43-.L26
	.long	.L43-.L26
	.long	.L28-.L26
	.long	.L43-.L26
	.long	.L43-.L26
	.long	.L43-.L26
	.long	.L43-.L26
	.long	.L43-.L26
	.long	.L27-.L26
	.long	.L43-.L26
	.long	.L25-.L26
	.text
.L29:
	call	dequeue
	movq	$13, -16(%rbp)
	jmp	.L34
.L25:
	movq	$13, -16(%rbp)
	jmp	.L34
.L32:
	movl	-28(%rbp), %eax
	cmpl	$4, %eax
	je	.L35
	cmpl	$4, %eax
	jg	.L36
	cmpl	$3, %eax
	je	.L37
	cmpl	$3, %eax
	jg	.L36
	cmpl	$1, %eax
	je	.L38
	cmpl	$2, %eax
	je	.L39
	jmp	.L36
.L35:
	movq	$3, -16(%rbp)
	jmp	.L40
.L37:
	movq	$0, -16(%rbp)
	jmp	.L40
.L39:
	movq	$4, -16(%rbp)
	jmp	.L40
.L38:
	movq	$7, -16(%rbp)
	jmp	.L40
.L36:
	movq	$2, -16(%rbp)
	nop
.L40:
	jmp	.L34
.L30:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, %edi
	call	exit@PLT
.L27:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$1, -16(%rbp)
	jmp	.L34
.L33:
	call	display
	movq	$13, -16(%rbp)
	jmp	.L34
.L28:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-24(%rbp), %eax
	movl	%eax, %edi
	call	enqueue
	movq	$13, -16(%rbp)
	jmp	.L34
.L31:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$13, -16(%rbp)
	jmp	.L34
.L43:
	nop
.L34:
	jmp	.L41
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.section	.rodata
.LC8:
	.string	"%d\n"
	.text
	.globl	display
	.type	display, @function
display:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$3, -8(%rbp)
.L59:
	cmpq	$7, -8(%rbp)
	ja	.L60
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L47(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L47(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L47:
	.long	.L52-.L47
	.long	.L60-.L47
	.long	.L51-.L47
	.long	.L50-.L47
	.long	.L61-.L47
	.long	.L60-.L47
	.long	.L48-.L47
	.long	.L46-.L47
	.text
.L50:
	movl	f(%rip), %eax
	cmpl	$-1, %eax
	jne	.L54
	movq	$2, -8(%rbp)
	jmp	.L56
.L54:
	movq	$6, -8(%rbp)
	jmp	.L56
.L48:
	movl	f(%rip), %eax
	movl	%eax, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L56
.L52:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	q(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L56
.L46:
	movl	r(%rip), %eax
	cmpl	%eax, -12(%rbp)
	jg	.L57
	movq	$0, -8(%rbp)
	jmp	.L56
.L57:
	movq	$4, -8(%rbp)
	jmp	.L56
.L51:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$4, -8(%rbp)
	jmp	.L56
.L60:
	nop
.L56:
	jmp	.L59
.L61:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	display, .-display
	.section	.rodata
.LC9:
	.string	"Overflow"
	.text
	.globl	enqueue
	.type	enqueue, @function
enqueue:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$5, -8(%rbp)
.L77:
	cmpq	$7, -8(%rbp)
	ja	.L78
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L65(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L65(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L65:
	.long	.L78-.L65
	.long	.L78-.L65
	.long	.L70-.L65
	.long	.L69-.L65
	.long	.L68-.L65
	.long	.L67-.L65
	.long	.L66-.L65
	.long	.L79-.L65
	.text
.L68:
	movl	r(%rip), %eax
	addl	$1, %eax
	movl	%eax, r(%rip)
	movl	f(%rip), %eax
	addl	$1, %eax
	movl	%eax, f(%rip)
	movl	r(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	leaq	q(%rip), %rdx
	movl	-20(%rbp), %eax
	movl	%eax, (%rcx,%rdx)
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$7, -8(%rbp)
	jmp	.L71
.L69:
	movl	r(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	q(%rip), %rax
	movl	(%rdx,%rax), %eax
	cmpl	$9, %eax
	jne	.L72
	movq	$6, -8(%rbp)
	jmp	.L71
.L72:
	movq	$2, -8(%rbp)
	jmp	.L71
.L66:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$7, -8(%rbp)
	jmp	.L71
.L67:
	movl	f(%rip), %eax
	cmpl	$-1, %eax
	jne	.L74
	movq	$4, -8(%rbp)
	jmp	.L71
.L74:
	movq	$3, -8(%rbp)
	jmp	.L71
.L70:
	movl	r(%rip), %eax
	addl	$1, %eax
	movl	%eax, r(%rip)
	movl	r(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	leaq	q(%rip), %rdx
	movl	-20(%rbp), %eax
	movl	%eax, (%rcx,%rdx)
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$7, -8(%rbp)
	jmp	.L71
.L78:
	nop
.L71:
	jmp	.L77
.L79:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	enqueue, .-enqueue
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
