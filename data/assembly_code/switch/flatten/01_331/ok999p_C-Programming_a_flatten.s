	.file	"ok999p_C-Programming_a_flatten.c"
	.text
	.globl	_TIG_IZ_CWCK_argv
	.bss
	.align 8
	.type	_TIG_IZ_CWCK_argv, @object
	.size	_TIG_IZ_CWCK_argv, 8
_TIG_IZ_CWCK_argv:
	.zero	8
	.globl	head
	.align 8
	.type	head, @object
	.size	head, 8
head:
	.zero	8
	.globl	capacity
	.align 4
	.type	capacity, @object
	.size	capacity, 4
capacity:
	.zero	4
	.globl	_TIG_IZ_CWCK_envp
	.align 8
	.type	_TIG_IZ_CWCK_envp, @object
	.size	_TIG_IZ_CWCK_envp, 8
_TIG_IZ_CWCK_envp:
	.zero	8
	.globl	new
	.align 8
	.type	new, @object
	.size	new, 8
new:
	.zero	8
	.globl	temp
	.align 8
	.type	temp, @object
	.size	temp, 8
temp:
	.zero	8
	.globl	_TIG_IZ_CWCK_argc
	.align 4
	.type	_TIG_IZ_CWCK_argc, @object
	.size	_TIG_IZ_CWCK_argc, 4
_TIG_IZ_CWCK_argc:
	.zero	4
	.globl	size
	.align 4
	.type	size, @object
	.size	size, 4
size:
	.zero	4
	.text
	.globl	create
	.type	create, @function
create:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$4, -16(%rbp)
.L15:
	cmpq	$6, -16(%rbp)
	ja	.L16
	movq	-16(%rbp), %rax
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
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L17-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L6:
	movq	$3, -16(%rbp)
	jmp	.L11
.L9:
	movq	head(%rip), %rax
	testq	%rax, %rax
	jne	.L12
	movq	$6, -16(%rbp)
	jmp	.L11
.L12:
	movq	$5, -16(%rbp)
	jmp	.L11
.L7:
	movl	$16, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, new(%rip)
	movq	new(%rip), %rax
	movl	-20(%rbp), %edx
	movl	%edx, (%rax)
	movq	new(%rip), %rax
	movq	$0, 8(%rax)
	movq	$1, -16(%rbp)
	jmp	.L11
.L3:
	movq	new(%rip), %rax
	movq	%rax, head(%rip)
	movq	$0, -16(%rbp)
	jmp	.L11
.L5:
	movq	new(%rip), %rax
	movq	head(%rip), %rdx
	movq	%rdx, 8(%rax)
	movq	$0, -16(%rbp)
	jmp	.L11
.L10:
	movq	new(%rip), %rax
	movq	%rax, head(%rip)
	movq	$2, -16(%rbp)
	jmp	.L11
.L16:
	nop
.L11:
	jmp	.L15
.L17:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	create, .-create
	.section	.rodata
.LC0:
	.string	"%d"
	.text
	.globl	peek
	.type	peek, @function
peek:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L23:
	cmpq	$0, -8(%rbp)
	je	.L19
	cmpq	$1, -8(%rbp)
	jne	.L25
	movq	head(%rip), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -8(%rbp)
	jmp	.L21
.L19:
	movl	$0, %eax
	jmp	.L24
.L25:
	nop
.L21:
	jmp	.L23
.L24:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	peek, .-peek
	.globl	pop
	.type	pop, @function
pop:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$0, -8(%rbp)
.L29:
	cmpq	$0, -8(%rbp)
	je	.L32
	nop
	jmp	.L29
.L32:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	pop, .-pop
	.section	.rodata
.LC1:
	.string	"Enter capacity: "
	.text
	.globl	main
	.type	main, @function
main:
.LFB8:
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
	movl	$0, size(%rip)
	nop
.L34:
	movl	$0, capacity(%rip)
	nop
.L35:
	movq	$0, temp(%rip)
	nop
.L36:
	movq	$0, head(%rip)
	nop
.L37:
	movq	$0, new(%rip)
	nop
.L38:
	movq	$0, _TIG_IZ_CWCK_envp(%rip)
	nop
.L39:
	movq	$0, _TIG_IZ_CWCK_argv(%rip)
	nop
.L40:
	movl	$0, _TIG_IZ_CWCK_argc(%rip)
	nop
	nop
.L41:
.L42:
#APP
# 181 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-CWCK--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_CWCK_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_CWCK_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_CWCK_envp(%rip)
	nop
	movq	$9, -16(%rbp)
.L55:
	cmpq	$11, -16(%rbp)
	ja	.L57
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L45(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L45(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L45:
	.long	.L50-.L45
	.long	.L57-.L45
	.long	.L57-.L45
	.long	.L57-.L45
	.long	.L57-.L45
	.long	.L57-.L45
	.long	.L49-.L45
	.long	.L57-.L45
	.long	.L48-.L45
	.long	.L47-.L45
	.long	.L46-.L45
	.long	.L44-.L45
	.text
.L48:
	movl	-24(%rbp), %eax
	cmpl	$1, %eax
	je	.L51
	cmpl	$2, %eax
	jne	.L52
	movq	$6, -16(%rbp)
	jmp	.L53
.L51:
	movq	$10, -16(%rbp)
	jmp	.L53
.L52:
	movq	$0, -16(%rbp)
	nop
.L53:
	jmp	.L54
.L44:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-28(%rbp), %eax
	movl	%eax, -20(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L54
.L47:
	movq	$11, -16(%rbp)
	jmp	.L54
.L49:
	call	peek
	movq	$0, -16(%rbp)
	jmp	.L54
.L46:
	call	push
	movq	$0, -16(%rbp)
	jmp	.L54
.L50:
	movl	$52, %edi
	call	putchar@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$8, -16(%rbp)
	jmp	.L54
.L57:
	nop
.L54:
	jmp	.L55
	.cfi_endproc
.LFE8:
	.size	main, .-main
	.section	.rodata
.LC2:
	.string	"Data: "
	.text
	.globl	push
	.type	push, @function
push:
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
	movq	$3, -16(%rbp)
.L67:
	cmpq	$3, -16(%rbp)
	je	.L59
	cmpq	$3, -16(%rbp)
	ja	.L70
	cmpq	$0, -16(%rbp)
	je	.L61
	cmpq	$2, -16(%rbp)
	je	.L71
	jmp	.L70
.L59:
	movl	capacity(%rip), %eax
	movl	size(%rip), %edx
	cmpl	%edx, %eax
	je	.L63
	movq	$0, -16(%rbp)
	jmp	.L65
.L63:
	movq	$2, -16(%rbp)
	jmp	.L65
.L61:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-20(%rbp), %eax
	movl	%eax, %edi
	call	create
	movl	size(%rip), %eax
	addl	$1, %eax
	movl	%eax, size(%rip)
	movq	$2, -16(%rbp)
	jmp	.L65
.L70:
	nop
.L65:
	jmp	.L67
.L71:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L69
	call	__stack_chk_fail@PLT
.L69:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	push, .-push
	.globl	isEmpty
	.type	isEmpty, @function
isEmpty:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$0, -8(%rbp)
.L75:
	cmpq	$0, -8(%rbp)
	jne	.L78
	movl	$0, %eax
	jmp	.L77
.L78:
	nop
	jmp	.L75
.L77:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	isEmpty, .-isEmpty
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
