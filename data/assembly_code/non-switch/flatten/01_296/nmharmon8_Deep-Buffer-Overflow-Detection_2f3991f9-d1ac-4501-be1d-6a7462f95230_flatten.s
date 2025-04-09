	.file	"nmharmon8_Deep-Buffer-Overflow-Detection_2f3991f9-d1ac-4501-be1d-6a7462f95230_flatten.c"
	.text
	.globl	_TIG_IZ_pc4M_envp
	.bss
	.align 8
	.type	_TIG_IZ_pc4M_envp, @object
	.size	_TIG_IZ_pc4M_envp, 8
_TIG_IZ_pc4M_envp:
	.zero	8
	.globl	_TIG_IZ_pc4M_argv
	.align 8
	.type	_TIG_IZ_pc4M_argv, @object
	.size	_TIG_IZ_pc4M_argv, 8
_TIG_IZ_pc4M_argv:
	.zero	8
	.globl	_TIG_IZ_pc4M_argc
	.align 4
	.type	_TIG_IZ_pc4M_argc, @object
	.size	_TIG_IZ_pc4M_argc, 4
_TIG_IZ_pc4M_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%d%d\n"
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
	subq	$256, %rsp
	movl	%edi, -228(%rbp)
	movq	%rsi, -240(%rbp)
	movq	%rdx, -248(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_pc4M_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_pc4M_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_pc4M_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 131 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-pc4M--0
# 0 "" 2
#NO_APP
	movl	-228(%rbp), %eax
	movl	%eax, _TIG_IZ_pc4M_argc(%rip)
	movq	-240(%rbp), %rax
	movq	%rax, _TIG_IZ_pc4M_argv(%rip)
	movq	-248(%rbp), %rax
	movq	%rax, _TIG_IZ_pc4M_envp(%rip)
	nop
	movq	$3, -200(%rbp)
.L18:
	cmpq	$8, -200(%rbp)
	ja	.L21
	movq	-200(%rbp), %rax
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
	.long	.L13-.L8
	.long	.L21-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L21-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L21-.L8
	.long	.L7-.L8
	.text
.L7:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L19
	jmp	.L20
.L11:
	movq	$2, -200(%rbp)
	jmp	.L15
.L9:
	cmpl	$0, -212(%rbp)
	js	.L16
	movq	$0, -200(%rbp)
	jmp	.L15
.L16:
	movq	$5, -200(%rbp)
	jmp	.L15
.L10:
	movl	-216(%rbp), %edx
	movl	-220(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -200(%rbp)
	jmp	.L15
.L13:
	movl	-212(%rbp), %eax
	cltq
	leaq	-112(%rbp), %rdx
	addq	%rdx, %rax
	movl	-212(%rbp), %edx
	movslq	%edx, %rdx
	leaq	-192(%rbp), %rcx
	addq	%rcx, %rdx
	movzbl	(%rax), %eax
	movb	%al, (%rdx)
	subl	$1, -212(%rbp)
	movq	$6, -200(%rbp)
	jmp	.L15
.L12:
	movl	$0, -208(%rbp)
	movl	$12, -204(%rbp)
	movl	$53, -220(%rbp)
	movl	$64, -216(%rbp)
	movl	-208(%rbp), %eax
	cltd
	idivl	-204(%rbp)
	movl	%eax, -220(%rbp)
	movl	-208(%rbp), %eax
	cltd
	idivl	-204(%rbp)
	movl	%eax, -216(%rbp)
	movl	-216(%rbp), %eax
	cltd
	idivl	-204(%rbp)
	movl	%edx, -216(%rbp)
	movl	-204(%rbp), %eax
	addl	%eax, -216(%rbp)
	subl	$1, -220(%rbp)
	movl	-220(%rbp), %eax
	imull	-208(%rbp), %eax
	movl	%eax, -220(%rbp)
	movl	$0, -212(%rbp)
	movq	$6, -200(%rbp)
	jmp	.L15
.L21:
	nop
.L15:
	jmp	.L18
.L20:
	call	__stack_chk_fail@PLT
.L19:
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
