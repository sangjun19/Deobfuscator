	.file	"Johnkb101313_learn_1-8_flatten.c"
	.text
	.globl	_TIG_IZ_f1iC_envp
	.bss
	.align 8
	.type	_TIG_IZ_f1iC_envp, @object
	.size	_TIG_IZ_f1iC_envp, 8
_TIG_IZ_f1iC_envp:
	.zero	8
	.globl	_TIG_IZ_f1iC_argc
	.align 4
	.type	_TIG_IZ_f1iC_argc, @object
	.size	_TIG_IZ_f1iC_argc, 4
_TIG_IZ_f1iC_argc:
	.zero	4
	.globl	_TIG_IZ_f1iC_argv
	.align 8
	.type	_TIG_IZ_f1iC_argv, @object
	.size	_TIG_IZ_f1iC_argv, 8
_TIG_IZ_f1iC_argv:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"space = %d\ntab = %d\nline = %d\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
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
	movq	$0, _TIG_IZ_f1iC_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_f1iC_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_f1iC_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 127 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-f1iC--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_f1iC_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_f1iC_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_f1iC_envp(%rip)
	nop
	movq	$3, -8(%rbp)
.L30:
	cmpq	$15, -8(%rbp)
	ja	.L31
	movq	-8(%rbp), %rax
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
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L31-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L31-.L8
	.long	.L31-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L31-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L15:
	movl	$0, -20(%rbp)
	movl	$0, -16(%rbp)
	movl	$0, -12(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L20
.L9:
	movl	$0, %eax
	jmp	.L21
.L7:
	movq	$10, -8(%rbp)
	jmp	.L20
.L18:
	addl	$1, -20(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L20
.L16:
	movq	$4, -8(%rbp)
	jmp	.L20
.L11:
	cmpl	$-1, -24(%rbp)
	je	.L22
	movq	$2, -8(%rbp)
	jmp	.L20
.L22:
	movq	$7, -8(%rbp)
	jmp	.L20
.L10:
	movl	-12(%rbp), %ecx
	movl	-16(%rbp), %edx
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -8(%rbp)
	jmp	.L20
.L14:
	addl	$1, -16(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L20
.L12:
	call	getchar@PLT
	movl	%eax, -24(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L20
.L19:
	addl	$1, -12(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L20
.L13:
	movl	$0, %eax
	jmp	.L21
.L17:
	cmpl	$101, -24(%rbp)
	je	.L24
	cmpl	$101, -24(%rbp)
	jg	.L25
	cmpl	$32, -24(%rbp)
	je	.L26
	cmpl	$32, -24(%rbp)
	jg	.L25
	cmpl	$9, -24(%rbp)
	je	.L27
	cmpl	$10, -24(%rbp)
	je	.L28
	jmp	.L25
.L24:
	movq	$13, -8(%rbp)
	jmp	.L29
.L28:
	movq	$0, -8(%rbp)
	jmp	.L29
.L27:
	movq	$6, -8(%rbp)
	jmp	.L29
.L26:
	movq	$1, -8(%rbp)
	jmp	.L29
.L25:
	movq	$15, -8(%rbp)
	nop
.L29:
	jmp	.L20
.L31:
	nop
.L20:
	jmp	.L30
.L21:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
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
