	.file	"rub3n-s_ISEC_traduz_flatten.c"
	.text
	.globl	_TIG_IZ_1OuG_argv
	.bss
	.align 8
	.type	_TIG_IZ_1OuG_argv, @object
	.size	_TIG_IZ_1OuG_argv, 8
_TIG_IZ_1OuG_argv:
	.zero	8
	.globl	_TIG_IZ_1OuG_argc
	.align 4
	.type	_TIG_IZ_1OuG_argc, @object
	.size	_TIG_IZ_1OuG_argc, 4
_TIG_IZ_1OuG_argc:
	.zero	4
	.globl	_TIG_IZ_1OuG_envp
	.align 8
	.type	_TIG_IZ_1OuG_envp, @object
	.size	_TIG_IZ_1OuG_envp, 8
_TIG_IZ_1OuG_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Palavra: "
.LC1:
	.string	"traducao [%s]\n"
.LC2:
	.string	"./rding"
.LC3:
	.string	"fork"
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
	addq	$-128, %rsp
	movl	%edi, -100(%rbp)
	movq	%rsi, -112(%rbp)
	movq	%rdx, -120(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_1OuG_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_1OuG_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_1OuG_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 108 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-1OuG--0
# 0 "" 2
#NO_APP
	movl	-100(%rbp), %eax
	movl	%eax, _TIG_IZ_1OuG_argc(%rip)
	movq	-112(%rbp), %rax
	movq	%rax, _TIG_IZ_1OuG_argv(%rip)
	movq	-120(%rbp), %rax
	movq	%rax, _TIG_IZ_1OuG_envp(%rip)
	nop
	movq	$3, -88(%rbp)
.L21:
	cmpq	$9, -88(%rbp)
	ja	.L24
	movq	-88(%rbp), %rax
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
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L24-.L8
	.long	.L9-.L8
	.long	.L24-.L8
	.long	.L7-.L8
	.text
.L11:
	movl	$-1, %eax
	jmp	.L22
.L14:
	cmpl	$-1, -96(%rbp)
	je	.L17
	cmpl	$0, -96(%rbp)
	jne	.L18
	movq	$0, -88(%rbp)
	jmp	.L19
.L17:
	movq	$7, -88(%rbp)
	jmp	.L19
.L18:
	movq	$9, -88(%rbp)
	nop
.L19:
	jmp	.L20
.L12:
	movq	$2, -88(%rbp)
	jmp	.L20
.L7:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-64(%rbp), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	gets@PLT
	movl	-80(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	-76(%rbp), %eax
	leaq	-64(%rbp), %rcx
	movl	$20, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movl	-68(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	-72(%rbp), %eax
	leaq	-32(%rbp), %rcx
	movl	$20, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	read@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -88(%rbp)
	jmp	.L20
.L10:
	movl	$0, %eax
	jmp	.L22
.L15:
	movl	$0, %edi
	call	close@PLT
	movl	-80(%rbp), %eax
	movl	%eax, %edi
	call	dup@PLT
	movl	-76(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	$1, %edi
	call	close@PLT
	movl	-68(%rbp), %eax
	movl	%eax, %edi
	call	dup@PLT
	movl	-72(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	-80(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	-68(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	$0, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	execl@PLT
	movq	$9, -88(%rbp)
	jmp	.L20
.L9:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$4, -88(%rbp)
	jmp	.L20
.L13:
	leaq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	pipe@PLT
	leaq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	pipe@PLT
	call	fork@PLT
	movl	%eax, -92(%rbp)
	movl	-92(%rbp), %eax
	movl	%eax, -96(%rbp)
	movq	$1, -88(%rbp)
	jmp	.L20
.L24:
	nop
.L20:
	jmp	.L21
.L22:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L23
	call	__stack_chk_fail@PLT
.L23:
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
