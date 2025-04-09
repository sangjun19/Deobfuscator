	.file	"AedenThomas_DSA_stacksArrayPractice_flatten.c"
	.text
	.globl	_TIG_IZ_1m4X_envp
	.bss
	.align 8
	.type	_TIG_IZ_1m4X_envp, @object
	.size	_TIG_IZ_1m4X_envp, 8
_TIG_IZ_1m4X_envp:
	.zero	8
	.globl	stack
	.align 32
	.type	stack, @object
	.size	stack, 400
stack:
	.zero	400
	.globl	i
	.align 4
	.type	i, @object
	.size	i, 4
i:
	.zero	4
	.globl	_TIG_IZ_1m4X_argv
	.align 8
	.type	_TIG_IZ_1m4X_argv, @object
	.size	_TIG_IZ_1m4X_argv, 8
_TIG_IZ_1m4X_argv:
	.zero	8
	.globl	n
	.align 4
	.type	n, @object
	.size	n, 4
n:
	.zero	4
	.globl	x
	.align 4
	.type	x, @object
	.size	x, 4
x:
	.zero	4
	.globl	top
	.align 4
	.type	top, @object
	.size	top, 4
top:
	.zero	4
	.globl	_TIG_IZ_1m4X_argc
	.align 4
	.type	_TIG_IZ_1m4X_argc, @object
	.size	_TIG_IZ_1m4X_argc, 4
_TIG_IZ_1m4X_argc:
	.zero	4
	.section	.rodata
	.align 8
.LC0:
	.string	"Enter the value to be pushed: "
.LC1:
	.string	"%d"
.LC2:
	.string	"Stack overflow!"
	.text
	.globl	push
	.type	push, @function
push:
.LFB2:
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
.L11:
	cmpq	$4, -16(%rbp)
	je	.L2
	cmpq	$4, -16(%rbp)
	ja	.L14
	cmpq	$3, -16(%rbp)
	je	.L4
	cmpq	$3, -16(%rbp)
	ja	.L14
	cmpq	$0, -16(%rbp)
	je	.L5
	cmpq	$1, -16(%rbp)
	je	.L15
	jmp	.L14
.L2:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	top(%rip), %eax
	addl	$1, %eax
	movl	%eax, top(%rip)
	movl	top(%rip), %edx
	movl	-20(%rbp), %eax
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	leaq	stack(%rip), %rdx
	movl	%eax, (%rcx,%rdx)
	movq	$1, -16(%rbp)
	jmp	.L7
.L4:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L7
.L5:
	movl	n(%rip), %eax
	leal	-1(%rax), %edx
	movl	top(%rip), %eax
	cmpl	%eax, %edx
	jg	.L9
	movq	$3, -16(%rbp)
	jmp	.L7
.L9:
	movq	$4, -16(%rbp)
	jmp	.L7
.L14:
	nop
.L7:
	jmp	.L11
.L15:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L13
	call	__stack_chk_fail@PLT
.L13:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	push, .-push
	.section	.rodata
.LC3:
	.string	"Enter size of stack: "
.LC4:
	.string	"\001. Push"
.LC5:
	.string	"\002. Pop"
.LC6:
	.string	"\003. Display"
.LC7:
	.string	"\nInvalid option"
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, i(%rip)
	nop
.L17:
	movl	$0, x(%rip)
	nop
.L18:
	movl	$-1, top(%rip)
	nop
.L19:
	movl	$0, n(%rip)
	nop
.L20:
	movl	$0, -20(%rbp)
	jmp	.L21
.L22:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	stack(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -20(%rbp)
.L21:
	cmpl	$99, -20(%rbp)
	jle	.L22
	nop
.L23:
	movq	$0, _TIG_IZ_1m4X_envp(%rip)
	nop
.L24:
	movq	$0, _TIG_IZ_1m4X_argv(%rip)
	nop
.L25:
	movl	$0, _TIG_IZ_1m4X_argc(%rip)
	nop
	nop
.L26:
.L27:
#APP
# 110 "AedenThomas_DSA_stacksArrayPractice.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-1m4X--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_1m4X_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_1m4X_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_1m4X_envp(%rip)
	nop
	movq	$9, -16(%rbp)
.L45:
	cmpq	$10, -16(%rbp)
	ja	.L48
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L30(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L30(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L30:
	.long	.L37-.L30
	.long	.L36-.L30
	.long	.L48-.L30
	.long	.L35-.L30
	.long	.L34-.L30
	.long	.L48-.L30
	.long	.L48-.L30
	.long	.L33-.L30
	.long	.L32-.L30
	.long	.L31-.L30
	.long	.L29-.L30
	.text
.L34:
	call	display
	movq	$8, -16(%rbp)
	jmp	.L38
.L32:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L46
	jmp	.L47
.L36:
	call	push
	movq	$8, -16(%rbp)
	jmp	.L38
.L35:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$0, -16(%rbp)
	jmp	.L38
.L31:
	movq	$3, -16(%rbp)
	jmp	.L38
.L29:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L38
.L37:
	movl	-28(%rbp), %eax
	cmpl	$3, %eax
	je	.L40
	cmpl	$3, %eax
	jg	.L41
	cmpl	$1, %eax
	je	.L42
	cmpl	$2, %eax
	je	.L43
	jmp	.L41
.L40:
	movq	$4, -16(%rbp)
	jmp	.L44
.L43:
	movq	$7, -16(%rbp)
	jmp	.L44
.L42:
	movq	$1, -16(%rbp)
	jmp	.L44
.L41:
	movq	$10, -16(%rbp)
	nop
.L44:
	jmp	.L38
.L33:
	call	pop
	movq	$8, -16(%rbp)
	jmp	.L38
.L48:
	nop
.L38:
	jmp	.L45
.L47:
	call	__stack_chk_fail@PLT
.L46:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main, .-main
	.section	.rodata
.LC8:
	.string	"\nStack Underflow"
.LC9:
	.string	"The popped element is %d "
	.text
	.globl	pop
	.type	pop, @function
pop:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L58:
	cmpq	$3, -8(%rbp)
	je	.L59
	cmpq	$3, -8(%rbp)
	ja	.L60
	cmpq	$2, -8(%rbp)
	je	.L52
	cmpq	$2, -8(%rbp)
	ja	.L60
	cmpq	$0, -8(%rbp)
	je	.L53
	cmpq	$1, -8(%rbp)
	jne	.L60
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L54
.L53:
	movl	top(%rip), %eax
	testl	%eax, %eax
	jns	.L56
	movq	$1, -8(%rbp)
	jmp	.L54
.L56:
	movq	$2, -8(%rbp)
	jmp	.L54
.L52:
	movl	top(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	stack(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	top(%rip), %eax
	subl	$1, %eax
	movl	%eax, top(%rip)
	movq	$3, -8(%rbp)
	jmp	.L54
.L60:
	nop
.L54:
	jmp	.L58
.L59:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	pop, .-pop
	.globl	display
	.type	display, @function
display:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$0, -8(%rbp)
.L64:
	cmpq	$0, -8(%rbp)
	je	.L67
	nop
	jmp	.L64
.L67:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	display, .-display
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
