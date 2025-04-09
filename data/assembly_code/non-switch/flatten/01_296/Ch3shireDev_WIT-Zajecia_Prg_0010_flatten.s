	.file	"Ch3shireDev_WIT-Zajecia_Prg_0010_flatten.c"
	.text
	.globl	_TIG_IZ_MbdV_argv
	.bss
	.align 8
	.type	_TIG_IZ_MbdV_argv, @object
	.size	_TIG_IZ_MbdV_argv, 8
_TIG_IZ_MbdV_argv:
	.zero	8
	.globl	_TIG_IZ_MbdV_envp
	.align 8
	.type	_TIG_IZ_MbdV_envp, @object
	.size	_TIG_IZ_MbdV_envp, 8
_TIG_IZ_MbdV_envp:
	.zero	8
	.globl	_TIG_IZ_MbdV_argc
	.align 4
	.type	_TIG_IZ_MbdV_argc, @object
	.size	_TIG_IZ_MbdV_argc, 4
_TIG_IZ_MbdV_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"T[%2d] ?="
.LC1:
	.string	"%d"
.LC2:
	.string	"\n\n\tMax[%d] = %d\n"
.LC3:
	.string	"size ?= "
.LC4:
	.string	"T[%2d] = %3d\n"
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
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_MbdV_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_MbdV_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_MbdV_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 107 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-MbdV--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_MbdV_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_MbdV_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_MbdV_envp(%rip)
	nop
	movq	$22, -24(%rbp)
.L32:
	cmpq	$22, -24(%rbp)
	ja	.L35
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L21-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L35-.L8
	.long	.L18-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L12:
	movl	-40(%rbp), %eax
	movl	%eax, -36(%rbp)
	movq	$5, -24(%rbp)
	jmp	.L22
.L20:
	movl	$0, -40(%rbp)
	movq	$11, -24(%rbp)
	jmp	.L22
.L15:
	movl	-40(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -40(%rbp)
	movq	$0, -24(%rbp)
	jmp	.L22
.L14:
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-32(%rbp), %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jg	.L23
	movq	$18, -24(%rbp)
	jmp	.L22
.L23:
	movq	$5, -24(%rbp)
	jmp	.L22
.L9:
	movl	-44(%rbp), %eax
	cmpl	%eax, -40(%rbp)
	jge	.L25
	movq	$16, -24(%rbp)
	jmp	.L22
.L25:
	movq	$4, -24(%rbp)
	jmp	.L22
.L16:
	movl	-44(%rbp), %eax
	cmpl	%eax, -40(%rbp)
	jge	.L27
	movq	$10, -24(%rbp)
	jmp	.L22
.L27:
	movq	$19, -24(%rbp)
	jmp	.L22
.L11:
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movl	-36(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$20, -24(%rbp)
	jmp	.L22
.L13:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-44(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-44(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -32(%rbp)
	movl	$0, -40(%rbp)
	movq	$0, -24(%rbp)
	jmp	.L22
.L7:
	movq	$17, -24(%rbp)
	jmp	.L22
.L19:
	addl	$1, -40(%rbp)
	movq	$21, -24(%rbp)
	jmp	.L22
.L17:
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movl	-40(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -40(%rbp)
	movq	$11, -24(%rbp)
	jmp	.L22
.L21:
	movl	-44(%rbp), %eax
	cmpl	%eax, -40(%rbp)
	jge	.L29
	movq	$12, -24(%rbp)
	jmp	.L22
.L29:
	movq	$7, -24(%rbp)
	jmp	.L22
.L18:
	movl	$1, -40(%rbp)
	movl	$0, -36(%rbp)
	movq	$21, -24(%rbp)
	jmp	.L22
.L10:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L33
	jmp	.L34
.L35:
	nop
.L22:
	jmp	.L32
.L34:
	call	__stack_chk_fail@PLT
.L33:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main, .-main
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
