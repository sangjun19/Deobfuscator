	.file	"mohammed-ali-boutaine_c-conditions1_5_flatten.c"
	.text
	.globl	_TIG_IZ_WeYn_argv
	.bss
	.align 8
	.type	_TIG_IZ_WeYn_argv, @object
	.size	_TIG_IZ_WeYn_argv, 8
_TIG_IZ_WeYn_argv:
	.zero	8
	.globl	_TIG_IZ_WeYn_envp
	.align 8
	.type	_TIG_IZ_WeYn_envp, @object
	.size	_TIG_IZ_WeYn_envp, 8
_TIG_IZ_WeYn_envp:
	.zero	8
	.globl	_TIG_IZ_WeYn_argc
	.align 4
	.type	_TIG_IZ_WeYn_argc, @object
	.size	_TIG_IZ_WeYn_argc, 4
_TIG_IZ_WeYn_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"entre un anne:"
.LC1:
	.string	"%d"
.LC2:
	.string	"chose un option:\n"
.LC3:
	.string	"1- Mois"
.LC4:
	.string	"2- Jours"
.LC5:
	.string	"3- Heures"
.LC6:
	.string	"4- Minutes"
.LC7:
	.string	"5- Secondes"
.LC8:
	.string	"entre votre choix:"
.LC9:
	.string	"%d an = %d s"
.LC10:
	.string	"%d an = %d mois"
.LC11:
	.string	"%d an = %d h"
.LC12:
	.string	"choix valide"
.LC13:
	.string	"%d an = %d min"
.LC14:
	.string	"%d an = %d jours"
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
	movq	$0, _TIG_IZ_WeYn_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_WeYn_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_WeYn_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 108 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-WeYn--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_WeYn_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_WeYn_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_WeYn_envp(%rip)
	nop
	movq	$9, -16(%rbp)
.L28:
	cmpq	$13, -16(%rbp)
	ja	.L31
	movq	-16(%rbp), %rax
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
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L31-.L8
	.long	.L14-.L8
	.long	.L31-.L8
	.long	.L31-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L31-.L8
	.long	.L7-.L8
	.text
.L14:
	movl	-20(%rbp), %eax
	cmpl	$5, %eax
	ja	.L18
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L20(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L20(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L20:
	.long	.L18-.L20
	.long	.L24-.L20
	.long	.L23-.L20
	.long	.L22-.L20
	.long	.L21-.L20
	.long	.L19-.L20
	.text
.L19:
	movq	$11, -16(%rbp)
	jmp	.L25
.L21:
	movq	$7, -16(%rbp)
	jmp	.L25
.L22:
	movq	$10, -16(%rbp)
	jmp	.L25
.L23:
	movq	$2, -16(%rbp)
	jmp	.L25
.L24:
	movq	$13, -16(%rbp)
	jmp	.L25
.L18:
	movq	$0, -16(%rbp)
	nop
.L25:
	jmp	.L26
.L12:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$4, -16(%rbp)
	jmp	.L26
.L16:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L29
	jmp	.L30
.L9:
	movl	-24(%rbp), %eax
	imull	$31536000, %eax, %edx
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L26
.L11:
	movq	$8, -16(%rbp)
	jmp	.L26
.L7:
	movl	-24(%rbp), %edx
	movl	%edx, %eax
	addl	%eax, %eax
	addl	%edx, %eax
	sall	$2, %eax
	movl	%eax, %edx
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L26
.L10:
	movl	-24(%rbp), %eax
	imull	$8760, %eax, %edx
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L26
.L17:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -16(%rbp)
	jmp	.L26
.L13:
	movl	-24(%rbp), %eax
	imull	$525600, %eax, %edx
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L26
.L15:
	movl	-24(%rbp), %eax
	imull	$365, %eax, %edx
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L26
.L31:
	nop
.L26:
	jmp	.L28
.L30:
	call	__stack_chk_fail@PLT
.L29:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
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
